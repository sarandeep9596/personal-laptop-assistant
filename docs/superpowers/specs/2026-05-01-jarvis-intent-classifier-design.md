# Jarvis Custom Intent Classifier — Design Spec

**Date:** 2026-05-01
**Status:** Approved (sections 1–5)
**Author:** Jatin (with brainstorming assist)
**Context:** College major project. Replace the Groq-hosted Decision-Making Model (DMM) in Jarvis with a self-trained DistilBERT classifier, integrated with a Groq fallback for low-confidence queries.

---

## 1. Goals

1. Demonstrate an end-to-end ML training pipeline (data → fine-tuning → evaluation → deployment) suitable for a college major-project viva.
2. Replace `Backend/Model.py::FirstLayerDMM` so it routes the **majority** of queries through a local model with **zero API cost**, falling back to Groq only on low-confidence cases.
3. Remove the user-visible pain of the 60-second rate-limit cooldown in `Main.py:83` for common queries.
4. Produce reportable artifacts: confusion matrix, per-class F1 table, latency comparison vs Groq, fallback-rate chart.

## 2. Non-Goals (explicit)

- Online / continuous learning from logged queries.
- NER-based argument extraction (rule-based extractor is sufficient for MVP).
- Multi-language input handling (the upstream `SpeechToText.py` already translates to English).
- MLOps / CI for retraining on new data.
- Any change to `Main.py`, the GUI, or downstream modules (`Automation.py`, `Chatbot.py`, `RealtimeSearchEngine.py`, etc.). The integration must be drop-in.

## 3. Architecture

**Base model:** `distilbert-base-uncased` (66M parameters, ~250MB). Chosen over TinyBERT/MobileBERT for the best accuracy-vs-explainability trade-off — examiner can be walked through every layer.

**Task formulation:** Single-label classification per clause. Multi-intent queries are split on conjunctions before classification.

**Hybrid local + Groq fallback:** local DistilBERT runs first; if max softmax confidence < `INTENT_CONFIDENCE_THRESHOLD` (default 0.6), or if the rule-based argument extractor returns `"unknown"`, the clause is forwarded to the existing Groq DMM as a fallback path.

**Intents (10 — taken verbatim from `Backend/Model.py:11`):**
`open`, `close`, `play`, `system`, `google search`, `youtube search`, `generate image`, `reminder`, `general`, `realtime`.

### 3.1 High-level flow

```
user query
   ↓
[ClauseSplitter.split]  →  N clauses
   ↓
for each clause:
   ↓
[IntentClassifier.predict]  →  (intent, confidence)
   ↓
   ├─ confidence ≥ THRESHOLD → [ArgumentExtractor.extract]
   │        ↓
   │        ├─ arg ≠ "unknown" → "{intent} {arg}"
   │        └─ arg == "unknown" → fallback to Groq DMM
   └─ confidence < THRESHOLD → fallback to Groq DMM
   ↓
[merge results] → ["open chrome", "google search …"]
   ↓
existing Main.py dispatch unchanged
```

### 3.2 Module layout

```
Backend/
├── ClauseSplitter.py        ← NEW (pure Python, ~10 LOC)
├── IntentClassifier.py      ← NEW (HF wrapper, ~40 LOC, lazy load)
├── ArgumentExtractor.py     ← NEW (rule-based dispatch, ~60 LOC)
└── Model.py                 ← MODIFIED (orchestrator + Groq fallback)
```

The original Groq logic in `Model.py` is preserved verbatim as `_groq_classify(prompt)` — the fallback path is byte-for-byte identical to today's behavior.

### 3.3 Public API (unchanged)

```python
def FirstLayerDMM(prompt: str = "test") -> List[str]:
    ...
```

`Main.py:192` does not change.

---

## 4. Data Strategy

**Target dataset:** ~3,000 labeled queries total, structured as **two disjoint sets**:
- **Held-out test set:** 150 queries — 15 raw seeds per intent × 10 intents. Never paraphrased, never augmented, never shown to the model in any form. Stored as `data/test.csv`.
- **Train+val pool:** ~2,850 queries (~285 per intent × 10) generated from the *other* seeds via paraphrasing + augmentation. Split 80/20 within this pool → train ~2,280 / val ~570. Stored as `data/intents.csv`.

This is the honest setup — the test set is held out **before** any synthetic data generation, so test queries cannot leak into training under any circumstance.

### 4.1 Three-stage pipeline (in this order)

**Stage 1 — Manual seeds (~1 hour):** 25 hand-crafted examples per intent (250 total). Cover phrasing, formality, word-order variation. Existing examples in `Backend/Model.py:17-95` are starting material. Saved to `data/seeds.csv`.

**Stage 2 — Hold out test set (immediate):** Randomly pick 15 of the 25 seeds per intent (with `random_seed=42`) → `data/test.csv` (150 rows). The remaining 10 seeds per intent (100 rows total) feed Stage 3.

**Stage 3 — LLM paraphrasing (~2 hours of Groq calls):** For each of the 100 remaining seeds, generate 25 paraphrases via Groq llama-3.3-70b with varied instructions (casual, with typos, polite/long, terse, reordered). Yields ~250 per intent (= ~2,500 rows).

**Stage 4 — Programmatic augmentation (~30 min):** Push to ~285/intent (= ~2,850 rows) with random casing, `nlpaug` keyboard-typo noise, punctuation variation. Saved as `data/intents.csv`.

### 4.2 Quality control (~2 hours, manual)

Skim final CSV; delete obvious nonsense paraphrases; fix mislabeled rows. Bad labels are the dominant source of small-classifier failure and the examiner will ask about data quality.

### 4.3 Class imbalance

`system` intent has only ~5 real underlying commands. Generate many phrasings of those 5 to fill the 300 quota (preferred over class-weight tricks — simpler, fewer surprises).

### 4.4 Reproducibility — committed to git

- `data/seeds.csv` (250 hand-written)
- `data/intents.csv` (~2,850 train+val pool)
- `data/test.csv` (150 raw held-out seeds — never seen by training)
- `scripts/augment.py` (Groq paraphrasing + nlpaug)
- Fixed `random_seed = 42` everywhere

---

## 5. Training Pipeline

### 5.1 Notebook

`notebooks/train_intent_classifier.ipynb`. Runs end-to-end from a fresh Colab T4 session. Mounts Drive for data and checkpoints.

**Stack:** HuggingFace `transformers` + `datasets` + `evaluate`, using the `Trainer` API (not raw PyTorch loops) — concise, defendable as the industry-standard fine-tuning workflow.

### 5.2 Hyperparameters

| Param | Value | Rationale |
|---|---|---|
| Base model | `distilbert-base-uncased` | 66M params; ~250MB |
| Max sequence length | 64 tokens | Covers >99% of voice queries |
| Batch size | 32 | Fits T4 16GB |
| Learning rate | 2e-5 | BERT-family default |
| Optimizer | AdamW | Standard |
| LR schedule | Linear warmup (10%) → decay | Standard |
| Epochs | 4 | More overfits on ~2.3K training samples |
| Weight decay | 0.01 | Mild regularization |
| Eval strategy | Every epoch, save best by macro-F1 | Standard |
| Loss | CrossEntropyLoss (no class weights) | Dataset is balanced post-Stage-4 |
| Random seed | 42 | Reproducibility |

Expected training time on T4: **15–25 minutes**.

### 5.3 Evaluation metrics (computed in the notebook)

1. Overall accuracy on test set (target ≥ 92%).
2. Macro-F1 (target ≥ 90%) — the headline metric in the report.
3. Per-class precision / recall / F1 (table in report).
4. Confusion matrix (heatmap PNG → `reports/confusion_matrix.png`).
5. Latency benchmark: average and p95 inference time on laptop CPU vs Groq round-trip.
6. Cost-saved estimate: queries-per-day × Groq cost-per-query → monthly $ saved.

### 5.4 Model export

Push to HuggingFace Hub: `model.push_to_hub("<username>/jarvis-intent-classifier")`. Loaded at inference via `AutoModelForSequenceClassification.from_pretrained(...)`. Free, professional-looking, clean URL for the report.

### 5.5 Optional optimization (recommended)

ONNX Runtime export + dynamic INT8 quantization. Final inference target: **<20ms on laptop CPU** (down from ~80ms PyTorch). Model size ~60MB (down from 250MB). Adds ~30 min of work; presented in report as "deployment optimization."

### 5.6 Notebook cell outline

1. Install + imports
2. Mount Drive, load `data/intents.csv` (train+val pool) and `data/test.csv` (held-out test)
3. Deterministic 80/20 train/val split of `intents.csv`; `test.csv` is the test set
4. Tokenize with `DistilBertTokenizerFast`
5. Build `Dataset` objects
6. `DistilBertForSequenceClassification(num_labels=10)`
7. Define `compute_metrics` (accuracy, macro-F1, per-class)
8. `Trainer(...).train()`
9. Evaluate on test, save confusion matrix PNG
10. CPU latency benchmark
11. `push_to_hub(...)`
12. (Optional) ONNX export + INT8 + re-benchmark

---

## 6. Integration

### 6.1 `Backend/ClauseSplitter.py`

```python
def split(query: str) -> List[str]:
    """Split on ' and ', ',', ' then ' (case-insensitive). Trim. Drop empties."""
```
Pure Python. ~10 LOC. No model.

### 6.2 `Backend/IntentClassifier.py`

- Lazy module-level load of HF model + tokenizer on first call (no startup cost when feature is unused).
- One method: `predict(text: str) -> Tuple[str, float]` returning `(intent, confidence)`.
- Reads `INTENT_MODEL_NAME` from `.env`.
- Forces CPU inference (`device='cpu'`).
- Tokenizer `max_length=64`, truncation enabled.
- ~40 LOC.

### 6.3 `Backend/ArgumentExtractor.py`

- One function: `extract(clause: str, intent: str) -> str`.
- Per-intent rules:
  - `open` / `close` / `play` — strip leading verb words (`open`, `please`, `can you`, `launch`, `start`, `fire up`, `boot`, `get me`).
  - `google search` — strip `google`, `search`, `for`.
  - `youtube search` — strip `youtube`, `search`, `for`.
  - `system` — lookup table:
    ```
    {"increase volume","volume up","louder","turn it up"}            → "volume up"
    {"decrease volume","volume down","quieter","turn it down"}       → "volume down"
    {"mute","mute the volume","silent","silence"}                    → "mute"
    {"unmute"}                                                       → "unmute"
    ```
    No match → return `"unknown"` (caller forwards to Groq).
  - `generate image` — strip `generate a photo/picture/image of`.
  - `reminder` — return verbatim (downstream parser is out of scope).
  - `general` / `realtime` — return verbatim.
- ~60 LOC including the strip-words list.

### 6.4 `Backend/Model.py` (modified — orchestrator)

```python
def FirstLayerDMM(prompt: str = "test") -> List[str]:
    if not USE_LOCAL_CLASSIFIER:
        return _groq_classify(prompt)            # original behavior

    clauses = ClauseSplitter.split(prompt)
    results = []
    for clause in clauses:
        intent, conf = IntentClassifier.predict(clause)

        if conf >= CONFIDENCE_THRESHOLD:
            arg = ArgumentExtractor.extract(clause, intent)
            if arg == "unknown":
                _log_prediction(clause, intent, conf, used_fallback=True)
                results.extend(_groq_classify(clause))
            else:
                _log_prediction(clause, intent, conf, used_fallback=False)
                results.append(f"{intent} {arg}".strip())
        else:
            _log_prediction(clause, intent, conf, used_fallback=True)
            results.extend(_groq_classify(clause))

    return results or [f"general {prompt}"]
```

`_groq_classify(prompt)` is the original `FirstLayerDMM` body, unchanged.

### 6.5 `.env` additions

```
INTENT_MODEL_NAME=<username>/jarvis-intent-classifier
INTENT_CONFIDENCE_THRESHOLD=0.6
USE_LOCAL_CLASSIFIER=true
```

`USE_LOCAL_CLASSIFIER=false` is the **demo-day kill switch**: flip it and Jarvis behaves exactly as it does today.

### 6.6 Observability

Logging helper lives **inside `Backend/Model.py`** as a small private function (no new module — keeps the responsibility close to where it's called):

```python
def _log_prediction(query: str, intent: str, conf: float, used_fallback: bool) -> None:
    """Append one JSONL row to Data/intent_log.jsonl. Best-effort; never raises."""
```

It opens the file in append mode each call (low write volume — at most a few per second, only during voice interactions). All exceptions are swallowed so logging can never break the user-facing path.

Append-only JSONL log at `Data/intent_log.jsonl`:
```json
{"ts":"2026-05-01T15:23:01","query":"open chrome","intent":"open","confidence":0.97,"used_fallback":false}
```

`scripts/analyze_log.py` reads the JSONL and produces:
- Fallback-rate-over-time line chart
- Confidence-distribution histogram
- Top-10 fallback-triggering queries (table; feeds next data round)

These charts go directly into the project report.

---

## 7. Testing & Evaluation

### 7.1 Layer 1 — Model evaluation (notebook, one-off)

Already covered in §5.3. Produces:
- `reports/confusion_matrix.png`
- `reports/per_class_metrics.csv`
- `reports/latency_comparison.csv`

### 7.2 Layer 2 — Code-level tests (`pytest`)

- `tests/test_clause_splitter.py` — table-driven, ~10 cases. Fast.
- `tests/test_argument_extractor.py` — table-driven, ~25 cases (3 per intent). Fast.
- `tests/test_intent_classifier_integration.py` — loads the real model, asserts intent prediction on ~20 known queries. Marked `@pytest.mark.slow`; skipped in default runs.

### 7.3 Layer 3 — End-to-end smoke test

`scripts/smoke_test.py`:
1. Loads model.
2. Runs 30 fixed demo queries through full `FirstLayerDMM`.
3. Prints table: `query | predicted | confidence | used_fallback`.
4. Runs once with `USE_LOCAL_CLASSIFIER=true` and once `=false`; prints side-by-side.

Run the morning of the viva as a sanity check.

### 7.4 Demo script (12 queries, scripted)

| # | Query | Demonstrates |
|---|---|---|
| 1 | `open chrome` | Single-intent, high confidence |
| 2 | `launch firefox for me please` | Phrasing robustness |
| 3 | `chrome please` | Truncation robustness |
| 4 | `mute` / `make it louder` | `system` lookup |
| 5 | `google who won the cricket world cup` | `google search` intent |
| 6 | `open chrome and play kesariya song` | Multi-intent (clause splitter) |
| 7 | `qwerty asdf zxcv` | Low confidence → graceful Groq fallback |
| 8 | `what is the temperature` | `realtime` intent |
| 9 | `tell me a joke` | `general` intent |
| 10 | `generate a photo of sunset over mountains` | `generate image` intent |
| 11 | (kill switch ON) `open chrome` | Identical pure-Groq path |
| 12 | (live) latency comparison script | Headline number for viva |

---

## 8. Deliverables

Committed to repo:
- `data/seeds.csv` (250 hand-written seeds)
- `data/intents.csv` (~2,850 paraphrased+augmented train+val pool)
- `data/test.csv` (150 raw held-out test seeds)
- `scripts/augment.py`, `scripts/smoke_test.py`, `scripts/analyze_log.py`
- `notebooks/train_intent_classifier.ipynb`
- `Backend/ClauseSplitter.py`, `Backend/IntentClassifier.py`, `Backend/ArgumentExtractor.py`
- Modified `Backend/Model.py`
- `tests/test_clause_splitter.py`, `tests/test_argument_extractor.py`, `tests/test_intent_classifier_integration.py`
- `reports/confusion_matrix.png`, `reports/per_class_metrics.csv`, `reports/latency_comparison.csv`
- Updated `Requirements.txt` (`transformers`, `torch`, `datasets`, `evaluate`, `nlpaug`, `huggingface_hub`, plus optional `onnxruntime`, `optimum`)

External:
- HuggingFace Hub repo: `<username>/jarvis-intent-classifier`

## 9. Out of scope (Future Work — for the report)

- Online learning from `Data/intent_log.jsonl`
- NER-based argument extraction (rule-based suffices for MVP)
- Multi-language input (handled upstream by `SpeechToText.py` translator)
- MLOps / CI retraining
- Replacing the Chatbot or RealtimeSearchEngine LLM calls (this project targets only the DMM)

## 10. Indicative timeline (~1 week part-time)

| Day | Work |
|---|---|
| 1 | Stage 1 seeds + Stage 2 paraphrasing script |
| 2 | Stage 3 augmentation + manual quality pass |
| 3 | Training notebook end-to-end; first model on Hub |
| 4 | Build `ClauseSplitter` / `IntentClassifier` / `ArgumentExtractor` + pytest |
| 5 | Wire into `Model.py`; smoke-test both kill-switch modes |
| 6 | (Optional) ONNX + INT8 optimization; re-benchmark |
| 7 | Generate report artifacts; write report; rehearse demo |

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Trained model accuracy is low (<85%) | Iterate on data: more seeds in weakest intents, re-paraphrase, raise confidence threshold so more queries go to Groq fallback. |
| Examiner asks "why not just use Groq?" | Latency, cost, offline capability, no rate-limit cooldown. Numbers in `reports/latency_comparison.csv` and the cost-saved estimate. |
| Model misbehaves during viva | `USE_LOCAL_CLASSIFIER=false` kill switch — instant revert to current behavior. |
| Argument extractor rules brittle on unseen phrasings | `"unknown"` return triggers Groq fallback for safety. Logged for next data round. |
| Multi-intent queries split incorrectly | Splitter rules are <10 LOC and table-tested; failing cases also fall back to Groq. |
