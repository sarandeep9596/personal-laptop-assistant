# Jarvis Custom Intent Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Groq-hosted Decision-Making Model in `Backend/Model.py` with a self-trained DistilBERT intent classifier, with the existing Groq path preserved as a low-confidence fallback.

**Architecture:** Per-clause `DistilBERT` (66M params) classification with rule-based argument extraction. Hybrid local + Groq fallback gated by softmax confidence and a `USE_LOCAL_CLASSIFIER` kill-switch. Three new pure-Python helper modules + one modified orchestrator (`Backend/Model.py`). Drop-in replacement — `FirstLayerDMM(prompt) -> List[str]` signature unchanged.

**Tech Stack:** Python 3.12, HuggingFace `transformers` + `datasets` + `evaluate`, PyTorch (CPU at inference), `nlpaug` for typo augmentation, `pytest` for testing, Google Colab T4 for training, HuggingFace Hub for model distribution.

**Spec:** `docs/superpowers/specs/2026-05-01-jarvis-intent-classifier-design.md`

---

## File Structure

**New files:**
- `data/seeds.csv` — 250 hand-written seed queries (committed)
- `data/intents.csv` — ~2,850 paraphrased+augmented train+val pool (committed)
- `data/test.csv` — 150 raw held-out test seeds (committed)
- `scripts/augment.py` — Stage 2 (Groq paraphrase) + Stage 3 (nlpaug) data pipeline
- `scripts/smoke_test.py` — end-to-end sanity check (both kill-switch modes)
- `scripts/analyze_log.py` — produces report charts from `Data/intent_log.jsonl`
- `notebooks/train_intent_classifier.ipynb` — Colab training notebook
- `Backend/ClauseSplitter.py` — pure-Python query splitter (~10 LOC)
- `Backend/IntentClassifier.py` — HF model wrapper, lazy-loaded (~40 LOC)
- `Backend/ArgumentExtractor.py` — rule-based extractor (~60 LOC)
- `tests/__init__.py`
- `tests/test_clause_splitter.py`
- `tests/test_argument_extractor.py`
- `tests/test_intent_classifier.py`
- `tests/test_model_orchestrator.py`
- `tests/test_intent_classifier_integration.py` (marked `@pytest.mark.slow`)
- `reports/.gitkeep`

**Modified files:**
- `Backend/Model.py` — orchestrator (lines 131–161 replaced; original logic preserved as `_groq_classify`)
- `Requirements.txt` — add ML deps
- `.gitignore` — add `reports/*.png`, `models/`, `*.onnx`, `Data/intent_log.jsonl` (already covered by `Data/`)
- `.env` — add 3 keys (developer copies values; don't commit)

**Notes on file boundaries:**
- `ClauseSplitter`, `IntentClassifier`, `ArgumentExtractor` are deliberately separate so each can be unit-tested in isolation. `Model.py` orchestrates the three plus the Groq fallback.
- `_log_prediction` lives inside `Backend/Model.py` (not its own file) — too small to deserve a module.

---

## Task 1: Project scaffolding

**Files:**
- Create: `data/`, `scripts/`, `notebooks/`, `tests/`, `reports/` directories
- Create: `tests/__init__.py`, `reports/.gitkeep`
- Modify: `Requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create directories and placeholder files**

```bash
mkdir -p data scripts notebooks tests reports
touch tests/__init__.py reports/.gitkeep
```

- [ ] **Step 2: Append new dependencies to Requirements.txt**

Open `Requirements.txt` and append (do not remove existing lines):

```
transformers>=4.46
torch>=2.4
datasets>=3.0
evaluate>=0.4
scikit-learn>=1.5
nlpaug>=1.1
huggingface_hub>=0.26
matplotlib>=3.9
seaborn>=0.13
pytest>=8.3
pandas>=2.2
```

- [ ] **Step 3: Update .gitignore**

Append to `.gitignore`:

```
reports/*.png
models/
*.onnx
__pycache__/
.pytest_cache/
```

(`Data/` is already ignored, which covers `Data/intent_log.jsonl`.)

- [ ] **Step 4: Install new deps in the existing venv**

Run: `.venv/bin/pip install -r Requirements.txt`
Expected: all packages install without error. (Torch download is ~750MB, takes a few minutes.)

- [ ] **Step 5: Verify pytest works**

Run: `.venv/bin/pytest --version`
Expected: `pytest 8.x.y`

- [ ] **Step 6: Commit**

```bash
git add Requirements.txt .gitignore tests/__init__.py reports/.gitkeep data/ scripts/ notebooks/
git commit -m "chore: scaffold dirs and add ML dependencies for intent classifier"
```

---

## Task 2: Hand-write seed dataset

**Files:**
- Create: `data/seeds.csv`

Goal: 25 seeds × 10 intents = 250 rows. Two columns: `query,intent`. CSV with header. UTF-8.

- [ ] **Step 1: Create `data/seeds.csv` with the header**

Create `data/seeds.csv` with first line:

```csv
query,intent
```

- [ ] **Step 2: Add 25 seeds for intent `open`**

Append to `data/seeds.csv`:

```csv
open chrome,open
open notepad,open
launch firefox,open
start spotify,open
open vscode please,open
fire up calculator,open
boot up word,open
get me chrome,open
chrome please,open
launch youtube app,open
open the file explorer,open
start zoom,open
open camera,open
launch slack,open
get teams open,open
open whatsapp,open
fire up edge,open
boot photoshop,open
open settings,open
launch terminal,open
open task manager,open
start discord,open
open the calculator app,open
launch outlook,open
open powerpoint,open
```

- [ ] **Step 3: Add 25 seeds for intent `close`**

Append:

```csv
close chrome,close
close notepad,close
shut down spotify,close
quit firefox,close
close vscode please,close
terminate calculator,close
close word,close
close camera,close
shut spotify,close
close the file explorer,close
quit zoom,close
close edge,close
shut down slack,close
close discord,close
quit teams,close
close whatsapp,close
shut photoshop,close
close settings,close
quit terminal,close
close outlook,close
shut down powerpoint,close
close excel,close
quit obsidian,close
close steam,close
shut down telegram,close
```

- [ ] **Step 4: Add 25 seeds for intent `play`**

Append:

```csv
play kesariya,play
play despacito,play
play arijit singh songs,play
play shape of you,play
play believer by imagine dragons,play
play perfect by ed sheeran,play
play tum hi ho,play
play raabta,play
play channa mereya,play
play closer chainsmokers,play
play ed sheeran perfect,play
play apna time aayega,play
play seedhe maut,play
play the night we met,play
play blinding lights,play
play levitating dua lipa,play
play satisfya,play
play yellow coldplay,play
play viva la vida,play
play see you again,play
play counting stars,play
play believer,play
play uptown funk,play
play bohemian rhapsody,play
play radioactive,play
```

- [ ] **Step 5: Add 25 seeds for intent `system`**

Append (cover only the 5 supported underlying actions, lots of phrasing per action):

```csv
increase volume,system
volume up,system
turn it up,system
louder,system
make it louder,system
decrease volume,system
volume down,system
turn it down,system
quieter,system
make it quieter,system
mute,system
mute the volume,system
silent,system
silence please,system
mute audio,system
unmute,system
unmute the volume,system
turn volume on,system
sound on,system
restore audio,system
crank the volume,system
bring the volume down,system
hush,system
shh,system
turn it off no wait turn it down,system
```

- [ ] **Step 6: Add 25 seeds for intent `google search`**

Append:

```csv
google who is virat kohli,google search
google who is ms dhoni,google search
google for who won the world cup,google search
search who is sachin tendulkar,google search
google what is python programming,google search
search the web for best laptops 2026,google search
google how does bert work,google search
search for indian recipes,google search
google for distance from earth to moon,google search
google modi government schemes,google search
search for jobs in bangalore,google search
google flight prices delhi to mumbai,google search
google reviews of iphone 17,google search
search the meaning of compunction,google search
google taylor swift latest album,google search
search for tutorials on react native,google search
google for chess openings,google search
search up isro upcoming missions,google search
google ipl 2026 schedule,google search
google for income tax slabs,google search
search recent news on ai,google search
google how to make pizza dough,google search
search for paris travel guide,google search
google capital of australia,google search
search the population of india,google search
```

- [ ] **Step 7: Add 25 seeds for intent `youtube search`**

Append:

```csv
search arijit singh on youtube,youtube search
youtube anuv jain songs,youtube search
search for python tutorials on youtube,youtube search
youtube cooking videos,youtube search
search ms dhoni interviews on youtube,youtube search
youtube workout videos for beginners,youtube search
search latest movie trailers on youtube,youtube search
youtube guitar tutorials,youtube search
search travel vlogs on youtube,youtube search
youtube tech reviews,youtube search
search lo-fi music on youtube,youtube search
youtube standup comedy,youtube search
search ted talks on youtube,youtube search
youtube history documentaries,youtube search
search for gaming videos on youtube,youtube search
youtube ml courses,youtube search
search live cricket commentary on youtube,youtube search
youtube cute puppy videos,youtube search
search asmr videos on youtube,youtube search
youtube physics lectures,youtube search
search bollywood dance covers on youtube,youtube search
youtube veg recipes,youtube search
search youtube for car reviews,youtube search
youtube ipl highlights,youtube search
search photography tutorials on youtube,youtube search
```

- [ ] **Step 8: Add 25 seeds for intent `generate image`**

Append:

```csv
generate a photo of sunset mountains,generate image
generate an image of a cat in a hat,generate image
create a picture of a dragon flying over a castle,generate image
generate a photo of a vintage car,generate image
make an image of a futuristic city,generate image
generate a picture of a samurai in cherry blossoms,generate image
create a photo of a beach at dawn,generate image
generate an image of a robot serving coffee,generate image
draw a picture of a forest in autumn,generate image
generate a photo of a spaceship landing on mars,generate image
create an image of an astronaut on a horse,generate image
generate a picture of a wizard casting a spell,generate image
make a photo of a moonlit lake,generate image
generate an image of a cyberpunk street market,generate image
create a picture of a phoenix rising,generate image
generate a photo of an old library,generate image
draw an image of a tiny dragon on a coffee cup,generate image
generate a picture of a lighthouse in a storm,generate image
make an image of a hot air balloon over fields,generate image
generate a photo of a misty mountain village,generate image
create a picture of a knight fighting a dragon,generate image
generate an image of a sloth wearing sunglasses,generate image
make a photo of an ancient temple in a jungle,generate image
generate a picture of a panda playing a guitar,generate image
create an image of a steampunk airship,generate image
```

- [ ] **Step 9: Add 25 seeds for intent `reminder`**

Append:

```csv
remind me to drink water at 8 pm,reminder
remind me to call mom at 7 pm,reminder
set a reminder for the meeting at 3,reminder
remind me about the dentist appointment tomorrow,reminder
remind me to take my medicine at 9 am,reminder
set a reminder to submit assignment at midnight,reminder
remind me to go for a run at 6 am,reminder
remind me to buy groceries this evening,reminder
set a reminder to pay the electricity bill,reminder
remind me about dad's birthday on saturday,reminder
remind me to study at 10 pm,reminder
set reminder for project review monday morning,reminder
remind me to feed the cat at 8 am,reminder
remind me to charge my phone tonight,reminder
set a reminder to renew my subscription,reminder
remind me to back up my code,reminder
remind me to check email at 9,reminder
set reminder to attend yoga class at 7 am,reminder
remind me to walk the dog,reminder
remind me to wish riya happy birthday,reminder
set a reminder for the flight at 5 am tomorrow,reminder
remind me to lock the door before sleeping,reminder
remind me to send the report by 5 pm,reminder
set a reminder to water the plants every monday,reminder
remind me to defrost the chicken,reminder
```

- [ ] **Step 10: Add 25 seeds for intent `general`**

Append:

```csv
how are you,general
do you like pizza,general
who is your creator,general
tell me a joke,general
what is your favourite color,general
how old are you,general
do you sleep,general
are you a robot,general
what do you think about love,general
do you have feelings,general
what is the meaning of life,general
say something nice,general
do you believe in god,general
who is sachin tendulkar,general
what is python,general
explain quantum computing simply,general
why is the sky blue,general
who wrote hamlet,general
what is recursion,general
explain newtons third law,general
what is photosynthesis,general
who painted the mona lisa,general
what is the largest planet,general
how do airplanes fly,general
what is artificial intelligence,general
```

- [ ] **Step 11: Add 25 seeds for intent `realtime`**

Append:

```csv
what is the temperature now,realtime
is today hot,realtime
will it rain today,realtime
how is the weather outside,realtime
is it cold outside,realtime
what is todays weather forecast,realtime
will i need an umbrella today,realtime
what is the humidity right now,realtime
tell me todays news,realtime
top news today,realtime
breaking news right now,realtime
current headlines,realtime
what is happening in the world today,realtime
what time is it,realtime
what is todays date,realtime
what day is it today,realtime
is it raining outside,realtime
how cold is it,realtime
news about cricket today,realtime
latest news in india,realtime
weather in mumbai now,realtime
will it snow today,realtime
how hot is it outside,realtime
read me the headlines,realtime
todays top stories,realtime
```

- [ ] **Step 12: Verify row count**

Run: `wc -l data/seeds.csv`
Expected: `251 data/seeds.csv` (header + 250 rows).

Run: `.venv/bin/python -c "import pandas as pd; df = pd.read_csv('data/seeds.csv'); print(df['intent'].value_counts())"`
Expected: each of the 10 intents shows count 25.

- [ ] **Step 13: Commit**

```bash
git add data/seeds.csv
git commit -m "data: add 250 hand-written seeds (25 per intent) for intent classifier"
```

---

## Task 3: Build the data augmentation pipeline

**Files:**
- Create: `scripts/augment.py`

The script does three things in order: (a) hold out 15 of 25 seeds per intent → `data/test.csv`; (b) Groq-paraphrase the remaining 10 seeds × 25 paraphrases each → ~250/intent; (c) `nlpaug` casing+typos to bring to ~285/intent → `data/intents.csv`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_augment_pipeline.py`:

```python
"""Smoke tests for the augmentation script's pure-Python pieces.
The Groq-call portion is not unit-tested here (network + cost); covered by
running the full pipeline manually in Task 4."""

import pandas as pd
from scripts.augment import holdout_test_set, augment_with_typos


def test_holdout_test_set_balanced(tmp_path):
    seeds = pd.DataFrame({
        "query": [f"q{i}" for i in range(50)],
        "intent": ["open"] * 25 + ["close"] * 25,
    })
    train_pool, test = holdout_test_set(seeds, n_test_per_intent=15, seed=42)
    assert len(test) == 30
    assert (test["intent"].value_counts() == 15).all()
    assert len(train_pool) == 20
    assert (train_pool["intent"].value_counts() == 10).all()
    # No leakage
    assert set(test["query"]).isdisjoint(set(train_pool["query"]))


def test_augment_with_typos_grows_dataset():
    df = pd.DataFrame({"query": ["open chrome"] * 10, "intent": ["open"] * 10})
    out = augment_with_typos(df, target_per_intent=15, seed=42)
    assert len(out) >= 15
    assert (out["intent"] == "open").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_augment_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.augment'`.

- [ ] **Step 3: Create `scripts/__init__.py`**

```bash
touch scripts/__init__.py
```

- [ ] **Step 4: Implement `scripts/augment.py`**

Create `scripts/augment.py`:

```python
"""Generate data/intents.csv (train+val pool) and data/test.csv (held-out test) from data/seeds.csv.

Pipeline:
  Stage 1 (manual): data/seeds.csv already exists (250 rows).
  Stage 2 (this script): hold out 15 seeds per intent → test.csv.
  Stage 3 (this script): Groq-paraphrase the remaining 10/intent × 25 paraphrases.
  Stage 4 (this script): nlpaug typo+casing → ~285/intent → intents.csv.
"""

from __future__ import annotations
import os
import random
import time
from pathlib import Path
import pandas as pd
from dotenv import dotenv_values

RANDOM_SEED = 42
N_TEST_PER_INTENT = 15
N_PARAPHRASES_PER_SEED = 25
TARGET_PER_INTENT = 285

ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = ROOT / "data" / "seeds.csv"
TEST_PATH = ROOT / "data" / "test.csv"
INTENTS_PATH = ROOT / "data" / "intents.csv"

PARAPHRASE_INSTRUCTIONS = [
    "Rephrase casually, as a teenager would say it.",
    "Rephrase with a small typo or grammar mistake (still understandable).",
    "Rephrase as a longer, more polite request.",
    "Rephrase as the shortest possible command.",
    "Rephrase using a different word order, keeping the meaning identical.",
]


def holdout_test_set(seeds: pd.DataFrame, n_test_per_intent: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_pool, test) where test has n_test_per_intent rows per intent and they are disjoint."""
    rng = random.Random(seed)
    test_rows, train_rows = [], []
    for intent, group in seeds.groupby("intent"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        test_idxs = set(idxs[:n_test_per_intent])
        for i in idxs:
            (test_rows if i in test_idxs else train_rows).append(seeds.loc[i])
    return (
        pd.DataFrame(train_rows).reset_index(drop=True),
        pd.DataFrame(test_rows).reset_index(drop=True),
    )


def paraphrase_with_groq(df: pd.DataFrame, n_per_seed: int) -> pd.DataFrame:
    """For each seed, generate n_per_seed paraphrases via Groq llama-3.3-70b. Skipped if no API key."""
    from groq import Groq
    env = dotenv_values(str(ROOT / ".env"))
    key = env.get("GroqAPIKey")
    if not key:
        raise RuntimeError("GroqAPIKey missing in .env — required for paraphrasing.")
    client = Groq(api_key=key)
    rng = random.Random(RANDOM_SEED)

    out_rows = []
    for _, row in df.iterrows():
        query, intent = row["query"], row["intent"]
        for j in range(n_per_seed):
            instr = rng.choice(PARAPHRASE_INSTRUCTIONS)
            prompt = (
                f"{instr}\n"
                f"Original: \"{query}\"\n"
                f"Output ONLY the rephrased sentence, no quotes, no explanation."
            )
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=40,
                        temperature=0.9,
                    )
                    text = resp.choices[0].message.content.strip().strip('"').strip("'")
                    if text:
                        out_rows.append({"query": text, "intent": intent})
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(20)
                        continue
                    print(f"[skip] {query!r} attempt {attempt}: {e}")
                    break
    return pd.DataFrame(out_rows)


def augment_with_typos(df: pd.DataFrame, target_per_intent: int, seed: int) -> pd.DataFrame:
    """Use nlpaug keyboard-typo + casing tricks to push each intent up to target_per_intent rows."""
    import nlpaug.augmenter.char as nac
    rng = random.Random(seed)
    keyboard_aug = nac.KeyboardAug(aug_char_min=1, aug_char_max=1, aug_word_p=0.1, include_special_char=False)

    grown = []
    for intent, group in df.groupby("intent"):
        rows = group.to_dict("records")
        while len(rows) < target_per_intent:
            base = rng.choice(group.to_dict("records"))
            q = base["query"]
            choice = rng.random()
            if choice < 0.4:
                q_new = q.upper()
            elif choice < 0.7:
                q_new = q.capitalize()
            else:
                try:
                    q_new = keyboard_aug.augment(q)[0]
                except Exception:
                    q_new = q
            rows.append({"query": q_new, "intent": intent})
        grown.extend(rows[:target_per_intent])
    return pd.DataFrame(grown)


def main():
    seeds = pd.read_csv(SEEDS_PATH)
    print(f"Loaded {len(seeds)} seeds from {SEEDS_PATH}")

    train_pool, test = holdout_test_set(seeds, N_TEST_PER_INTENT, RANDOM_SEED)
    test.to_csv(TEST_PATH, index=False)
    print(f"Wrote {len(test)} held-out test rows to {TEST_PATH}")
    print(f"Train pool seeds: {len(train_pool)} (10 per intent)")

    print("Paraphrasing via Groq — this takes ~30 min for 100 seeds × 25 paraphrases…")
    paraphrased = paraphrase_with_groq(train_pool, N_PARAPHRASES_PER_SEED)
    combined = pd.concat([train_pool, paraphrased], ignore_index=True)
    print(f"Post-paraphrase: {len(combined)} rows")

    final = augment_with_typos(combined, TARGET_PER_INTENT, RANDOM_SEED)
    final = final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    final.to_csv(INTENTS_PATH, index=False)
    print(f"Wrote {len(final)} train+val rows to {INTENTS_PATH}")
    print(final["intent"].value_counts())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run unit tests**

Run: `.venv/bin/pytest tests/test_augment_pipeline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/augment.py tests/test_augment_pipeline.py
git commit -m "feat(data): add augmentation pipeline (holdout, Groq paraphrase, nlpaug typos)"
```

---

## Task 4: Run the data pipeline (manual; produces `data/intents.csv` + `data/test.csv`)

**Files:**
- Generated: `data/intents.csv`, `data/test.csv`

This task is **manual / one-shot** because it calls Groq (requires `.env` with `GroqAPIKey` and ~30 min of API time). The output is committed to git so the rest of the project is reproducible without re-running it.

- [ ] **Step 1: Confirm `.env` has `GroqAPIKey`**

Run: `grep '^GroqAPIKey' .env`
Expected: a line with the key (not committed).

- [ ] **Step 2: Run the pipeline**

Run: `.venv/bin/python scripts/augment.py`
Expected stdout:
- `Loaded 250 seeds from .../data/seeds.csv`
- `Wrote 150 held-out test rows to .../data/test.csv`
- `Train pool seeds: 100 (10 per intent)`
- `Paraphrasing via Groq…` (then a long pause)
- `Post-paraphrase: ~2,600 rows` (some calls may fail and be skipped — anything ≥ 2,300 is fine)
- `Wrote 2850 train+val rows to .../data/intents.csv`
- value_counts showing 285 per intent.

If you hit a hard 429 burst, the script will sleep 20s and retry up to 3 times per call.

- [ ] **Step 3: Spot-check the output (manual quality pass — ~30 min)**

Open `data/intents.csv` in a viewer / spreadsheet. Skim through:
- Look for obvious garbage paraphrases (e.g., LLM ignored instruction and answered the query).
- Look for mislabeled rows.
- Delete bad rows directly.

Don't aim for perfection — kill the worst 5–10% of rows. Good enough is good enough.

- [ ] **Step 4: Verify final state**

Run: `.venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/intents.csv')['intent'].value_counts())"`
Expected: each intent has ≥ 250 rows (after manual deletion).

Run: `.venv/bin/python -c "import pandas as pd; t=pd.read_csv('data/test.csv'); i=pd.read_csv('data/intents.csv'); leak=set(t['query']) & set(i['query']); print('LEAKED:', leak)"`
Expected: `LEAKED: set()` — no test queries appear in the train pool.

- [ ] **Step 5: Commit**

```bash
git add data/intents.csv data/test.csv
git commit -m "data: generate intents.csv (train+val pool) and test.csv (held-out)"
```

---

## Task 5: Write the training notebook

**Files:**
- Create: `notebooks/train_intent_classifier.ipynb`

The notebook is structured so a Colab user can mount Drive (containing `data/intents.csv` and `data/test.csv`) and run all cells top-to-bottom. The cells below should be one cell per bullet block.

- [ ] **Step 1: Create the notebook file via JSON**

Create `notebooks/train_intent_classifier.ipynb` with this exact JSON content (paste verbatim):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Jarvis Intent Classifier Training\n\nFine-tune `distilbert-base-uncased` on `data/intents.csv` (train+val) and evaluate on `data/test.csv`.\n\n**Runtime:** Colab T4 GPU. Expected training time: 15–25 minutes."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!pip install -q transformers datasets evaluate scikit-learn huggingface_hub matplotlib seaborn"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import os, json, random, time\nimport numpy as np, pandas as pd, torch\nimport matplotlib.pyplot as plt, seaborn as sns\nfrom datasets import Dataset\nfrom transformers import (\n    AutoTokenizer, AutoModelForSequenceClassification,\n    TrainingArguments, Trainer, DataCollatorWithPadding,\n)\nimport evaluate\nfrom sklearn.metrics import classification_report, confusion_matrix\n\nSEED = 42\nrandom.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\nprint('CUDA available:', torch.cuda.is_available())"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["from google.colab import drive\ndrive.mount('/content/drive')\n# Adjust this path to where you put your data folder in Drive:\nDATA_DIR = '/content/drive/MyDrive/jarvis-intent/data'\nOUT_DIR = '/content/drive/MyDrive/jarvis-intent/out'\nos.makedirs(OUT_DIR, exist_ok=True)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["intents_df = pd.read_csv(f'{DATA_DIR}/intents.csv')\ntest_df    = pd.read_csv(f'{DATA_DIR}/test.csv')\nprint('train+val pool:', len(intents_df))\nprint('test         :', len(test_df))\n\nLABELS = sorted(intents_df['intent'].unique().tolist())\nLABEL2ID = {l:i for i,l in enumerate(LABELS)}\nID2LABEL = {i:l for l,i in LABEL2ID.items()}\nprint('labels:', LABELS)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["intents_df = intents_df.sample(frac=1, random_state=SEED).reset_index(drop=True)\nsplit = int(0.8 * len(intents_df))\ntrain_df = intents_df.iloc[:split].copy()\nval_df   = intents_df.iloc[split:].copy()\nfor df in (train_df, val_df, test_df):\n    df['label'] = df['intent'].map(LABEL2ID)\nprint('train', len(train_df), 'val', len(val_df), 'test', len(test_df))"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["MODEL = 'distilbert-base-uncased'\ntok = AutoTokenizer.from_pretrained(MODEL)\n\ndef tokenize(batch):\n    return tok(batch['query'], truncation=True, max_length=64)\n\ntrain_ds = Dataset.from_pandas(train_df[['query','label']]).map(tokenize, batched=True)\nval_ds   = Dataset.from_pandas(val_df  [['query','label']]).map(tokenize, batched=True)\ntest_ds  = Dataset.from_pandas(test_df [['query','label']]).map(tokenize, batched=True)\ncollator = DataCollatorWithPadding(tok)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["model = AutoModelForSequenceClassification.from_pretrained(\n    MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID\n)\n\nacc = evaluate.load('accuracy')\nf1  = evaluate.load('f1')\n\ndef compute_metrics(pred):\n    preds = pred.predictions.argmax(-1)\n    return {\n        'accuracy': acc.compute(predictions=preds, references=pred.label_ids)['accuracy'],\n        'macro_f1': f1.compute(predictions=preds, references=pred.label_ids, average='macro')['f1'],\n    }"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["args = TrainingArguments(\n    output_dir=f'{OUT_DIR}/checkpoints',\n    num_train_epochs=4,\n    per_device_train_batch_size=32,\n    per_device_eval_batch_size=64,\n    learning_rate=2e-5,\n    weight_decay=0.01,\n    warmup_ratio=0.1,\n    eval_strategy='epoch',\n    save_strategy='epoch',\n    load_best_model_at_end=True,\n    metric_for_best_model='macro_f1',\n    greater_is_better=True,\n    logging_steps=20,\n    seed=SEED,\n    report_to='none',\n)\n\ntrainer = Trainer(\n    model=model, args=args,\n    train_dataset=train_ds, eval_dataset=val_ds,\n    tokenizer=tok, data_collator=collator,\n    compute_metrics=compute_metrics,\n)\ntrainer.train()"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["pred = trainer.predict(test_ds)\npreds = pred.predictions.argmax(-1)\ntrue  = pred.label_ids\nprint(f\"Test accuracy: {(preds == true).mean():.4f}\")\nprint(classification_report(true, preds, target_names=LABELS, digits=4))\n\nreport_dict = classification_report(true, preds, target_names=LABELS, output_dict=True)\npd.DataFrame(report_dict).T.to_csv(f'{OUT_DIR}/per_class_metrics.csv')\n\ncm = confusion_matrix(true, preds)\nplt.figure(figsize=(10,8))\nsns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d', cmap='Blues')\nplt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion matrix (test set)')\nplt.tight_layout()\nplt.savefig(f'{OUT_DIR}/confusion_matrix.png', dpi=150)\nplt.show()"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import time\nmodel_cpu = trainer.model.to('cpu').eval()\nsample_queries = test_df['query'].sample(50, random_state=SEED).tolist()\n\nlatencies = []\nfor q in sample_queries:\n    enc = tok(q, return_tensors='pt', truncation=True, max_length=64)\n    t0 = time.perf_counter()\n    with torch.no_grad():\n        _ = model_cpu(**enc)\n    latencies.append((time.perf_counter() - t0) * 1000)\n\nimport numpy as np\nprint(f'Local CPU latency (ms) — mean: {np.mean(latencies):.1f}  p95: {np.percentile(latencies,95):.1f}')\npd.DataFrame({'cpu_ms': latencies}).describe().to_csv(f'{OUT_DIR}/latency_local.csv')"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["from huggingface_hub import login\nlogin()  # paste a write-scope token from huggingface.co/settings/tokens\n\nHUB_REPO = 'YOUR-HF-USERNAME/jarvis-intent-classifier'  # ← change this\ntrainer.model.push_to_hub(HUB_REPO)\ntok.push_to_hub(HUB_REPO)\nprint('Pushed to', HUB_REPO)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## (Optional) ONNX + INT8 quantization"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!pip install -q optimum[onnxruntime] onnxruntime\nfrom optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer\nfrom optimum.onnxruntime.configuration import AutoQuantizationConfig\n\nort_dir = f'{OUT_DIR}/onnx'\nort_model = ORTModelForSequenceClassification.from_pretrained(HUB_REPO, export=True)\nort_model.save_pretrained(ort_dir)\ntok.save_pretrained(ort_dir)\n\nquantizer = ORTQuantizer.from_pretrained(ort_dir)\nqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)\nquantizer.quantize(save_dir=f'{OUT_DIR}/onnx-int8', quantization_config=qconfig)\nprint('Quantized model saved to', f'{OUT_DIR}/onnx-int8')"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Validate notebook JSON**

Run: `.venv/bin/python -c "import json; json.load(open('notebooks/train_intent_classifier.ipynb'))" && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/train_intent_classifier.ipynb
git commit -m "feat(train): add Colab notebook for DistilBERT intent classifier"
```

---

## Task 6: Train and publish the model (manual — Colab session)

**Files:** none in repo. Outputs land on HuggingFace Hub.

- [ ] **Step 1: Upload data + notebook to Drive**

Copy `data/intents.csv` and `data/test.csv` to a Drive folder, e.g. `MyDrive/jarvis-intent/data/`.
Copy `notebooks/train_intent_classifier.ipynb` to a Drive folder you can open in Colab.

- [ ] **Step 2: Open notebook in Colab; set runtime to GPU**

Runtime → Change runtime type → T4 GPU.

- [ ] **Step 3: Edit `HUB_REPO` cell**

Replace `YOUR-HF-USERNAME` with your actual HF username (or any namespace where you have write access). Example: `jatin-roop/jarvis-intent-classifier`.

- [ ] **Step 4: Run all cells**

Runtime → Run all. Expected wall-clock: ~25 min.

- [ ] **Step 5: Verify outputs**

After completion:
- Test accuracy printed should be ≥ 0.92.
- Confusion matrix PNG saved to `MyDrive/jarvis-intent/out/`.
- Model visible at `https://huggingface.co/<your-username>/jarvis-intent-classifier`.

If accuracy < 0.85, do **not** ship — go back to Task 4 Step 3 and add more seeds for the worst-performing intents (visible in the per-class report), then re-run Tasks 4–6.

- [ ] **Step 6: Download report artifacts into the repo**

Download `confusion_matrix.png`, `per_class_metrics.csv`, `latency_local.csv` from Drive into local `reports/`.

```bash
ls reports/
# expect: confusion_matrix.png  per_class_metrics.csv  latency_local.csv
```

- [ ] **Step 7: Commit**

```bash
git add reports/per_class_metrics.csv reports/latency_local.csv
git commit -m "report: add training-run metrics and CPU latency"
# confusion_matrix.png is gitignored; keep it locally for the viva slide
```

---

## Task 7: Build `Backend/ClauseSplitter.py` (TDD)

**Files:**
- Create: `Backend/ClauseSplitter.py`
- Create: `tests/test_clause_splitter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_clause_splitter.py`:

```python
import pytest
from Backend.ClauseSplitter import split


@pytest.mark.parametrize("query,expected", [
    ("open chrome", ["open chrome"]),
    ("", []),
    ("   ", []),
    ("open chrome and play kesariya", ["open chrome", "play kesariya"]),
    ("open chrome and play kesariya then mute", ["open chrome", "play kesariya", "mute"]),
    ("open chrome, then mute", ["open chrome", "mute"]),
    ("OPEN CHROME AND PLAY KESARIYA", ["OPEN CHROME", "PLAY KESARIYA"]),
    ("open chrome   and    play kesariya", ["open chrome", "play kesariya"]),
    ("a, b, c, d", ["a", "b", "c", "d"]),
    ("standalone", ["standalone"]),
])
def test_split(query, expected):
    assert split(query) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clause_splitter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `Backend/ClauseSplitter.py`**

Create `Backend/ClauseSplitter.py`:

```python
"""Split a natural-language query into single-intent clauses."""
from __future__ import annotations
import re
from typing import List

_SPLIT_RE = re.compile(r"\s*(?:,|\band\b|\bthen\b)\s*", flags=re.IGNORECASE)


def split(query: str) -> List[str]:
    if not query or not query.strip():
        return []
    parts = _SPLIT_RE.split(query.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clause_splitter.py -v`
Expected: PASS (10 cases).

- [ ] **Step 5: Commit**

```bash
git add Backend/ClauseSplitter.py tests/test_clause_splitter.py
git commit -m "feat(intent): add ClauseSplitter for multi-intent queries"
```

---

## Task 8: Build `Backend/ArgumentExtractor.py` (TDD)

**Files:**
- Create: `Backend/ArgumentExtractor.py`
- Create: `tests/test_argument_extractor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_argument_extractor.py`:

```python
import pytest
from Backend.ArgumentExtractor import extract


@pytest.mark.parametrize("intent,clause,expected", [
    # open
    ("open", "open chrome", "chrome"),
    ("open", "please open notepad", "notepad"),
    ("open", "launch firefox for me please", "firefox"),
    # close
    ("close", "close chrome", "chrome"),
    ("close", "shut down spotify", "spotify"),
    ("close", "quit firefox please", "firefox"),
    # play
    ("play", "play kesariya", "kesariya"),
    ("play", "play arijit singh songs", "arijit singh songs"),
    ("play", "play despacito please", "despacito"),
    # google search
    ("google search", "google who is dhoni", "who is dhoni"),
    ("google search", "search for python tutorials", "python tutorials"),
    ("google search", "google for ipl 2026 schedule", "ipl 2026 schedule"),
    # youtube search
    ("youtube search", "youtube arijit singh", "arijit singh"),
    ("youtube search", "search lo-fi music on youtube", "lo-fi music"),
    ("youtube search", "search for ted talks on youtube", "ted talks"),
    # system — known phrases
    ("system", "increase volume", "volume up"),
    ("system", "louder", "volume up"),
    ("system", "decrease volume", "volume down"),
    ("system", "quieter", "volume down"),
    ("system", "mute", "mute"),
    ("system", "mute the volume", "mute"),
    ("system", "unmute", "unmute"),
    # system — unknown phrase falls through
    ("system", "turn off the dishwasher", "unknown"),
    # generate image
    ("generate image", "generate a photo of sunset mountains", "sunset mountains"),
    ("generate image", "create an image of a dragon", "a dragon"),
    ("generate image", "make a picture of a robot", "a robot"),
    # reminder + general + realtime — verbatim passthrough
    ("reminder", "remind me to call mom at 7", "remind me to call mom at 7"),
    ("general", "tell me a joke", "tell me a joke"),
    ("realtime", "what is the temperature", "what is the temperature"),
])
def test_extract(intent, clause, expected):
    assert extract(clause, intent) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_argument_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `Backend/ArgumentExtractor.py`**

Create `Backend/ArgumentExtractor.py`:

```python
"""Per-intent rule-based extractor: clause + intent → argument string.
Returns 'unknown' for system commands that don't match any known phrase."""
from __future__ import annotations
import re

_LEADING_FILLERS = (
    "please", "can you", "could you", "would you", "kindly",
    "for me", "now", "right now", "go ahead and",
)

_OPEN_VERBS = ("open", "launch", "start", "fire up", "boot up", "boot", "get me", "fire")
_CLOSE_VERBS = ("close", "shut down", "shut", "quit", "terminate", "stop", "kill", "exit")
_PLAY_VERBS = ("play",)
_GOOGLE_LEADERS = ("google", "search the web for", "search the web", "search for", "search", "for", "up", "on google")
_YOUTUBE_LEADERS = ("youtube", "search for", "search", "for", "on youtube", "up")
_IMAGE_LEADERS = (
    "generate a photo of", "generate an image of", "generate a picture of",
    "create a photo of", "create an image of", "create a picture of",
    "make a photo of", "make an image of", "make a picture of",
    "draw a picture of", "draw an image of", "draw",
    "generate", "create", "make",
)

_SYSTEM_LOOKUP = {
    "volume up": {
        "increase volume", "volume up", "louder", "make it louder",
        "turn it up", "crank the volume", "sound up", "raise the volume",
    },
    "volume down": {
        "decrease volume", "volume down", "quieter", "make it quieter",
        "turn it down", "bring the volume down", "lower the volume",
    },
    "mute": {
        "mute", "mute the volume", "silent", "silence", "silence please",
        "mute audio", "hush", "shh",
    },
    "unmute": {
        "unmute", "unmute the volume", "turn volume on", "sound on", "restore audio",
    },
}


def _strip_leaders(text: str, leaders: tuple[str, ...]) -> str:
    out = text.strip().lower()
    changed = True
    while changed:
        changed = False
        for lead in sorted(leaders, key=len, reverse=True):
            pat = r"^" + re.escape(lead) + r"\b[\s,.:;!?-]*"
            new = re.sub(pat, "", out, count=1)
            if new != out:
                out = new
                changed = True
        for filler in _LEADING_FILLERS:
            pat = r"^" + re.escape(filler) + r"\b[\s,.:;!?-]*"
            new = re.sub(pat, "", out, count=1)
            if new != out:
                out = new
                changed = True
    return re.sub(r"\s+", " ", out).strip()


def _strip_trailing_fillers(text: str) -> str:
    out = text
    changed = True
    while changed:
        changed = False
        for filler in _LEADING_FILLERS:
            pat = r"[\s,.:;!?-]*\b" + re.escape(filler) + r"\s*$"
            new = re.sub(pat, "", out, count=1)
            if new != out:
                out = new
                changed = True
    return out.strip()


def _system_lookup(clause: str) -> str:
    norm = re.sub(r"\s+", " ", clause.strip().lower())
    for canonical, phrases in _SYSTEM_LOOKUP.items():
        if norm in phrases:
            return canonical
    return "unknown"


def extract(clause: str, intent: str) -> str:
    if intent in ("reminder", "general", "realtime"):
        return clause.strip()

    if intent == "system":
        return _system_lookup(clause)

    if intent == "open":
        return _strip_trailing_fillers(_strip_leaders(clause, _OPEN_VERBS))
    if intent == "close":
        return _strip_trailing_fillers(_strip_leaders(clause, _CLOSE_VERBS))
    if intent == "play":
        return _strip_trailing_fillers(_strip_leaders(clause, _PLAY_VERBS))
    if intent == "google search":
        return _strip_trailing_fillers(_strip_leaders(clause, _GOOGLE_LEADERS))
    if intent == "youtube search":
        return _strip_trailing_fillers(_strip_leaders(clause, _YOUTUBE_LEADERS))
    if intent == "generate image":
        return _strip_trailing_fillers(_strip_leaders(clause, _IMAGE_LEADERS))

    return clause.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_argument_extractor.py -v`
Expected: PASS (28 cases). If any fail, add the failing leader phrase to the relevant tuple at the top of the file.

- [ ] **Step 5: Commit**

```bash
git add Backend/ArgumentExtractor.py tests/test_argument_extractor.py
git commit -m "feat(intent): add ArgumentExtractor with per-intent rules + system lookup"
```

---

## Task 9: Build `Backend/IntentClassifier.py` (TDD with mock + slow integration test)

**Files:**
- Create: `Backend/IntentClassifier.py`
- Create: `tests/test_intent_classifier.py` (fast, mocked)
- Create: `tests/test_intent_classifier_integration.py` (slow, loads real model)
- Modify: `pyproject.toml` or `pytest.ini` to register the `slow` marker — we'll create `pytest.ini` from scratch.

- [ ] **Step 1: Create `pytest.ini` to register the `slow` marker**

Create `pytest.ini` at the repo root:

```ini
[pytest]
markers =
    slow: requires the trained HF model to be downloaded (skipped by default)
addopts = -m "not slow"
```

- [ ] **Step 2: Write the failing fast test**

Create `tests/test_intent_classifier.py`:

```python
"""Fast unit tests for IntentClassifier — HF model is mocked."""
from __future__ import annotations
import importlib
from unittest.mock import patch, MagicMock
import numpy as np

import Backend.IntentClassifier as ic


def _fresh_module():
    """Reload module so the lazy cache is reset between tests."""
    importlib.reload(ic)
    return ic


def test_lazy_load_called_only_once():
    mod = _fresh_module()
    fake_model = MagicMock()
    fake_tok = MagicMock()
    fake_model.config.id2label = {0: "open", 1: "close"}

    with patch.object(mod, "_load_model", return_value=(fake_model, fake_tok)) as loader:
        # _predict_logits is also patched so we don't need a real forward pass
        with patch.object(mod, "_predict_logits", return_value=np.array([2.0, 0.0])):
            mod.predict("open chrome")
            mod.predict("close notepad")
        assert loader.call_count == 1


def test_predict_returns_intent_and_confidence():
    mod = _fresh_module()
    fake_model = MagicMock()
    fake_model.config.id2label = {0: "open", 1: "close"}
    with patch.object(mod, "_load_model", return_value=(fake_model, MagicMock())):
        with patch.object(mod, "_predict_logits", return_value=np.array([5.0, 0.0])):
            intent, conf = mod.predict("open chrome")
    assert intent == "open"
    assert 0.99 <= conf <= 1.0  # softmax(5,0) ≈ 0.993
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_intent_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'Backend.IntentClassifier'`.

- [ ] **Step 4: Implement `Backend/IntentClassifier.py`**

Create `Backend/IntentClassifier.py`:

```python
"""Lazy-loaded DistilBERT intent classifier wrapper.

Reads INTENT_MODEL_NAME from .env; defaults to a placeholder so import never
fails during testing. Forces CPU inference."""
from __future__ import annotations
import os
from typing import Tuple
import numpy as np
from dotenv import dotenv_values

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env = dotenv_values(os.path.join(_BASE_DIR, ".env"))
MODEL_NAME = _env.get("INTENT_MODEL_NAME", "distilbert-base-uncased")
MAX_LEN = 64

_model = None
_tokenizer = None


def _load_model():
    """Heavy import + download, called once on first predict()."""
    import torch  # local import keeps Backend importable without torch installed
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cpu").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tok


def _predict_logits(text: str) -> np.ndarray:
    import torch
    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = _model(**enc).logits.squeeze(0).numpy()
    return logits


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def predict(text: str) -> Tuple[str, float]:
    """Return (intent_label, max_softmax_confidence)."""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = _load_model()
    logits = _predict_logits(text)
    probs = _softmax(logits)
    idx = int(probs.argmax())
    label = _model.config.id2label[idx]
    return label, float(probs[idx])
```

- [ ] **Step 5: Run fast tests to verify they pass**

Run: `.venv/bin/pytest tests/test_intent_classifier.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Write the slow integration test**

Create `tests/test_intent_classifier_integration.py`:

```python
"""Slow integration tests — load the real model from HF Hub."""
import pytest
import Backend.IntentClassifier as ic


@pytest.mark.slow
@pytest.mark.parametrize("text,expected", [
    ("open chrome", "open"),
    ("close firefox", "close"),
    ("play kesariya", "play"),
    ("google who is dhoni", "google search"),
    ("youtube arijit singh", "youtube search"),
    ("mute the volume", "system"),
    ("generate a photo of mountains", "generate image"),
    ("remind me to call mom", "reminder"),
    ("tell me a joke", "general"),
    ("what is the temperature now", "realtime"),
])
def test_real_model_predicts_known_queries(text, expected):
    intent, conf = ic.predict(text)
    assert intent == expected, f"got {intent!r} (conf={conf:.2f}) for {text!r}"
    assert conf > 0.5
```

- [ ] **Step 7: Run slow integration test (only after Task 6 has published the model)**

Run: `.venv/bin/pytest tests/test_intent_classifier_integration.py -v -m slow`
Expected: PASS (10 tests). Skipped by default in normal `pytest` runs.

- [ ] **Step 8: Commit**

```bash
git add pytest.ini Backend/IntentClassifier.py tests/test_intent_classifier.py tests/test_intent_classifier_integration.py
git commit -m "feat(intent): add IntentClassifier wrapper with lazy load + tests"
```

---

## Task 10: Wire orchestrator into `Backend/Model.py`

**Files:**
- Modify: `Backend/Model.py`
- Create: `tests/test_model_orchestrator.py`

The strategy: rename the current body of `FirstLayerDMM` into `_groq_classify`, then write the new `FirstLayerDMM` that calls the local pipeline with Groq fallback. Add `_log_prediction` helper.

- [ ] **Step 1: Write the failing orchestrator test**

Create `tests/test_model_orchestrator.py`:

```python
"""Orchestrator tests — IntentClassifier, ArgumentExtractor, and Groq are all mocked."""
from __future__ import annotations
import importlib
from unittest.mock import patch
import os
import pytest


def _fresh():
    import Backend.Model as M
    importlib.reload(M)
    return M


@pytest.fixture(autouse=True)
def stub_env(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_LOCAL_CLASSIFIER", "true")
    monkeypatch.setenv("INTENT_CONFIDENCE_THRESHOLD", "0.6")
    # Redirect log file to tmp so tests don't pollute Data/
    monkeypatch.setenv("INTENT_LOG_PATH", str(tmp_path / "intent_log.jsonl"))


def test_high_confidence_local_path():
    M = _fresh()
    with patch("Backend.IntentClassifier.predict", return_value=("open", 0.95)), \
         patch("Backend.ArgumentExtractor.extract", return_value="chrome"), \
         patch.object(M, "_groq_classify") as groq:
        out = M.FirstLayerDMM("open chrome")
    assert out == ["open chrome"]
    groq.assert_not_called()


def test_low_confidence_falls_back_to_groq():
    M = _fresh()
    with patch("Backend.IntentClassifier.predict", return_value=("open", 0.3)), \
         patch.object(M, "_groq_classify", return_value=["general open chrome"]):
        out = M.FirstLayerDMM("xqv asdf")
    assert out == ["general open chrome"]


def test_unknown_argument_falls_back_to_groq():
    M = _fresh()
    with patch("Backend.IntentClassifier.predict", return_value=("system", 0.99)), \
         patch("Backend.ArgumentExtractor.extract", return_value="unknown"), \
         patch.object(M, "_groq_classify", return_value=["general turn off dishwasher"]):
        out = M.FirstLayerDMM("turn off the dishwasher")
    assert out == ["general turn off dishwasher"]


def test_multi_clause_dispatch():
    M = _fresh()
    def predict(text):
        return {"open chrome": ("open", 0.95),
                "play kesariya": ("play", 0.97)}[text]
    def extract(clause, intent):
        return {"open chrome": "chrome", "play kesariya": "kesariya"}[clause]
    with patch("Backend.IntentClassifier.predict", side_effect=predict), \
         patch("Backend.ArgumentExtractor.extract", side_effect=extract), \
         patch.object(M, "_groq_classify") as groq:
        out = M.FirstLayerDMM("open chrome and play kesariya")
    assert out == ["open chrome", "play kesariya"]
    groq.assert_not_called()


def test_kill_switch_routes_directly_to_groq(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_CLASSIFIER", "false")
    M = _fresh()
    with patch.object(M, "_groq_classify", return_value=["general hi"]) as groq, \
         patch("Backend.IntentClassifier.predict") as predict:
        out = M.FirstLayerDMM("hi")
    assert out == ["general hi"]
    predict.assert_not_called()
    groq.assert_called_once()


def test_empty_query_returns_general_fallback():
    M = _fresh()
    with patch("Backend.IntentClassifier.predict") as predict, \
         patch.object(M, "_groq_classify"):
        out = M.FirstLayerDMM("")
    assert out == ["general "]
    predict.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_model_orchestrator.py -v`
Expected: FAIL — current `Backend/Model.py` has no `_groq_classify`, no clause splitting, etc. Failures expected on every test.

- [ ] **Step 3: Replace `Backend/Model.py` with the new orchestrator**

Open `Backend/Model.py` and **replace the entire file** with:

```python
"""First-Layer Decision-Making Model.

Hybrid pipeline: local DistilBERT (via Backend.IntentClassifier) + rule-based
argument extraction (Backend.ArgumentExtractor), with the original Groq
classifier preserved as a low-confidence fallback (_groq_classify).

USE_LOCAL_CLASSIFIER=false in .env routes everything to Groq (kill switch)."""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List

from dotenv import dotenv_values
from groq import Groq

from Backend import ClauseSplitter, IntentClassifier, ArgumentExtractor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_vars = dotenv_values(os.path.join(BASE_DIR, ".env"))
GroqAPIKey = env_vars.get("GroqAPIKey", "")

USE_LOCAL_CLASSIFIER = os.environ.get(
    "USE_LOCAL_CLASSIFIER", env_vars.get("USE_LOCAL_CLASSIFIER", "true")
).strip().lower() == "true"
CONFIDENCE_THRESHOLD = float(os.environ.get(
    "INTENT_CONFIDENCE_THRESHOLD", env_vars.get("INTENT_CONFIDENCE_THRESHOLD", "0.6")
))
INTENT_LOG_PATH = os.environ.get(
    "INTENT_LOG_PATH", os.path.join(BASE_DIR, "Data", "intent_log.jsonl")
)

client = Groq(api_key=GroqAPIKey) if GroqAPIKey else None

# ----------------------------------------------------------------------------
# Original Groq DMM — preserved verbatim as the fallback path.
# ----------------------------------------------------------------------------

_FUNCS = [
    "exit", "general", "realtime", "open", "close", "play",
    "generate image", "system", "content", "google search",
    "youtube search", "reminder",
]

_CHAT_HISTORY = [
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "general how are you"},
    {"role": "user", "content": "open chrome"},
    {"role": "assistant", "content": "open chrome"},
    {"role": "user", "content": "close chrome"},
    {"role": "assistant", "content": "close chrome"},
    {"role": "user", "content": "google who is ms dhoni"},
    {"role": "assistant", "content": "google search who is ms dhoni"},
    {"role": "user", "content": "search arjit singh on youtube"},
    {"role": "assistant", "content": "youtube search arjit singh"},
    {"role": "user", "content": "play kesariya song"},
    {"role": "assistant", "content": "play kesariya song"},
    {"role": "user", "content": "increase volume"},
    {"role": "assistant", "content": "system volume up"},
    {"role": "user", "content": "remind me to study at 8 pm"},
    {"role": "assistant", "content": "reminder 8 pm study"},
    {"role": "user", "content": "generate a photo of a sunset mountain"},
    {"role": "assistant", "content": "generate image sunset mountain"},
    {"role": "user", "content": "what is temperature now"},
    {"role": "assistant", "content": "realtime what is temperature now"},
    {"role": "user", "content": "tell me today's news"},
    {"role": "assistant", "content": "realtime today's news"},
    {"role": "user", "content": "open chrome and search for mahadev quotes"},
    {"role": "assistant", "content": "open chrome, google search mahadev quotes"},
]

_PREAMBLE = """
You are a Decision-Making Model (DMM). You NEVER answer questions.
You only classify queries into the correct task type.

Rules:
- "open X" → open
- "close X" → close
- "play X" → play
- "google ..." → google search
- "youtube ..." → youtube search
- "remind me" → reminder
- "generate a picture/photo/image" → generate image
- "increase volume / mute" → system
- "news", "headlines", "breaking news", "top news" → realtime

Realtime queries are:
weather, temperature, humidity, rainfall, forecast, climate, sunrise, sunset,
"is today hot", "is it raining", "how cold is outside", today's date, time.

General queries are:
normal chatting / opinions / history / no automation.

If multiple tasks, output them separated by commas.
If unclear, classify as: general (query).

OUTPUT MUST FOLLOW THIS EXACT FORMAT:
open chrome
google search virat kohli
play kesariya
general who is sachin
realtime what is temperature now
"""


def _groq_classify(prompt: str) -> List[str]:
    """The original FirstLayerDMM body, untouched. Used as the fallback path."""
    if client is None:
        return [f"general {prompt}"]
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": _PREAMBLE}, *_CHAT_HISTORY,
                  {"role": "user", "content": prompt}],
        stream=True, temperature=0.2, max_tokens=80,
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            result += chunk.choices[0].delta.content
    result = result.replace("\n", "")
    parts = [p.strip() for p in result.split(",")]
    filtered = [t for t in parts if any(t.startswith(f) for f in _FUNCS)]
    return filtered or [f"general {prompt}"]


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def _log_prediction(query: str, intent: str, conf: float, used_fallback: bool) -> None:
    """Append-only JSONL log; best-effort, never raises."""
    try:
        os.makedirs(os.path.dirname(INTENT_LOG_PATH), exist_ok=True)
        line = json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "intent": intent,
            "confidence": round(conf, 4),
            "used_fallback": used_fallback,
        })
        with open(INTENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Public API — drop-in replacement, signature unchanged.
# ----------------------------------------------------------------------------

def FirstLayerDMM(prompt: str = "test") -> List[str]:
    if not prompt or not prompt.strip():
        return [f"general {prompt}"]

    if not USE_LOCAL_CLASSIFIER:
        return _groq_classify(prompt)

    clauses = ClauseSplitter.split(prompt)
    if not clauses:
        return [f"general {prompt}"]

    results: List[str] = []
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


if __name__ == "__main__":
    while True:
        q = input(">>> ")
        print(FirstLayerDMM(q))
```

- [ ] **Step 4: Add `Backend/__init__.py` if it doesn't exist**

Run: `ls Backend/__init__.py 2>/dev/null || touch Backend/__init__.py`
Expected: empty file is fine. (Without this, `from Backend import ClauseSplitter` won't resolve as a package import.)

- [ ] **Step 5: Run all fast tests**

Run: `.venv/bin/pytest -v`
Expected: ALL PASS — splitter (10) + extractor (28) + classifier (2) + orchestrator (6) + augment (2) = **48 tests** passing. Slow tests skipped.

- [ ] **Step 6: Commit**

```bash
git add Backend/__init__.py Backend/Model.py tests/test_model_orchestrator.py
git commit -m "feat(intent): wire local-first orchestrator into Backend/Model.py"
```

---

## Task 11: Document `.env` additions

**Files:**
- Create: `.env.example` (committed; the actual `.env` stays gitignored)

- [ ] **Step 1: Create `.env.example`**

Create `.env.example`:

```
# Existing keys (required)
Username=YourName
Assistantname=Jarvis
GroqAPIKey=
GNEWS_API=
HuggingFaceAPIKey=
AssistantVoice=en-US-GuyNeural
InputLanguage=en

# NEW — intent classifier
INTENT_MODEL_NAME=YOUR-HF-USERNAME/jarvis-intent-classifier
INTENT_CONFIDENCE_THRESHOLD=0.6
USE_LOCAL_CLASSIFIER=true
```

- [ ] **Step 2: Update your local `.env`**

Add the three new keys to your local `.env` (set `INTENT_MODEL_NAME` to the actual HF Hub repo published in Task 6).

- [ ] **Step 3: Commit `.env.example` only**

```bash
git add .env.example
git commit -m "docs: document .env keys including intent classifier config"
```

---

## Task 12: Smoke-test script

**Files:**
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Implement `scripts/smoke_test.py`**

Create `scripts/smoke_test.py`:

```python
"""Run 30 demo queries through FirstLayerDMM in both kill-switch modes and
print results side-by-side. Intended as the morning-of-viva sanity check."""
from __future__ import annotations
import os
import sys
import importlib

DEMO_QUERIES = [
    "open chrome",
    "launch firefox for me please",
    "chrome please",
    "close notepad",
    "shut down spotify",
    "play kesariya",
    "play arijit singh songs",
    "mute",
    "make it louder",
    "increase volume",
    "google who won the world cup",
    "search for python tutorials",
    "youtube arijit singh",
    "search lo-fi music on youtube",
    "open chrome and play kesariya",
    "open notepad then mute",
    "what is the temperature",
    "is it raining today",
    "tell me todays news",
    "tell me a joke",
    "who is sachin tendulkar",
    "generate a photo of sunset over mountains",
    "remind me to call mom at 7",
    "what time is it",
    "qwerty asdf zxcv",
    "open chrome and play kesariya and mute",
    "google ipl 2026 schedule",
    "youtube workout videos",
    "shut down chrome",
    "fire up vscode",
]


def run(use_local: bool):
    os.environ["USE_LOCAL_CLASSIFIER"] = "true" if use_local else "false"
    if "Backend.Model" in sys.modules:
        importlib.reload(sys.modules["Backend.Model"])
    from Backend.Model import FirstLayerDMM
    out = {}
    for q in DEMO_QUERIES:
        try:
            out[q] = FirstLayerDMM(q)
        except Exception as e:
            out[q] = [f"ERROR: {e}"]
    return out


def main():
    print("Running with USE_LOCAL_CLASSIFIER=true …")
    local = run(use_local=True)
    print("Running with USE_LOCAL_CLASSIFIER=false …")
    groq = run(use_local=False)

    print(f"\n{'QUERY':<55} {'LOCAL':<40} {'GROQ':<40}")
    print("-" * 135)
    for q in DEMO_QUERIES:
        l = ", ".join(local[q])[:38]
        g = ", ".join(groq[q])[:38]
        marker = "  " if l == g else " *"
        print(f"{q[:53]:<55} {l:<40} {g:<40}{marker}")
    print("\n* = local and groq disagreed on this query")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (manual — needs a real .env + HF model)**

Run: `.venv/bin/python -m scripts.smoke_test`
Expected: a table comparing local vs Groq output for 30 queries. Most should match; differences are interesting and may inform another data round.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat(scripts): add smoke_test for both kill-switch modes"
```

---

## Task 13: Log-analysis script for the report

**Files:**
- Create: `scripts/analyze_log.py`

- [ ] **Step 1: Implement `scripts/analyze_log.py`**

Create `scripts/analyze_log.py`:

```python
"""Read Data/intent_log.jsonl and produce three artifacts in reports/:
  - reports/fallback_rate.png — fallback rate over time (per 50-query bucket)
  - reports/confidence_hist.png — confidence distribution
  - reports/top_fallbacks.csv — top 10 fallback-triggering queries"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "Data" / "intent_log.jsonl"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def load_log(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    if not LOG_PATH.exists():
        print(f"No log at {LOG_PATH} — run Jarvis with USE_LOCAL_CLASSIFIER=true first.")
        return
    df = load_log(LOG_PATH)
    print(f"Loaded {len(df)} rows")

    # Fallback rate over time, in 50-query rolling buckets
    df["used_fallback"] = df["used_fallback"].astype(int)
    df["fallback_rate"] = df["used_fallback"].rolling(window=50, min_periods=10).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["fallback_rate"])
    plt.xlabel("query #"); plt.ylabel("fallback rate (50-query rolling)")
    plt.title("Groq-fallback rate over time")
    plt.tight_layout()
    plt.savefig(OUT / "fallback_rate.png", dpi=150)
    plt.close()

    # Confidence histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df["confidence"], bins=30)
    plt.xlabel("max softmax confidence"); plt.ylabel("count")
    plt.title("Confidence distribution across all queries")
    plt.tight_layout()
    plt.savefig(OUT / "confidence_hist.png", dpi=150)
    plt.close()

    # Top fallback-triggering queries
    falls = df[df["used_fallback"] == 1]
    top = falls["query"].value_counts().head(10).reset_index()
    top.columns = ["query", "fallback_count"]
    top.to_csv(OUT / "top_fallbacks.csv", index=False)

    print("Wrote:")
    for p in [OUT / "fallback_rate.png", OUT / "confidence_hist.png", OUT / "top_fallbacks.csv"]:
        print(" ", p)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/analyze_log.py
git commit -m "feat(scripts): add log-analysis for report charts"
```

---

## Task 14 (OPTIONAL): ONNX + INT8 quantization

**Files:** notebook cells already present in `notebooks/train_intent_classifier.ipynb` (the last cell). This task is just running them.

- [ ] **Step 1: In Colab, run the optional ONNX cell**

After Task 6 has trained and pushed the model, run the final notebook cell (`pip install optimum[onnxruntime]` + `ORTModelForSequenceClassification` + quantizer). It saves to `MyDrive/jarvis-intent/out/onnx-int8/`.

- [ ] **Step 2: Push the quantized ONNX model to a separate HF Hub branch (or repo)**

In Colab, after quantization:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=f'{OUT_DIR}/onnx-int8',
    repo_id=HUB_REPO,
    path_in_repo='onnx-int8',
)
```

- [ ] **Step 3: Update `Backend/IntentClassifier.py` to optionally use ONNX**

This is **optional** and only worth doing if you want the latency win for the viva. If you skip it, mention "ONNX path implemented but not wired" in the report's Future Work section.

If you do wire it, the change is small — replace `_load_model` with:

```python
def _load_model():
    use_onnx = os.environ.get("USE_ONNX", "false").lower() == "true"
    if use_onnx:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME, subfolder="onnx-int8")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="onnx-int8")
        return model, tok
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cpu").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tok
```

- [ ] **Step 4: Add `USE_ONNX=true` to `.env`, re-run smoke test, capture latency**

Re-run `.venv/bin/python -m scripts.smoke_test` and capture timings; add to `reports/latency_local.csv`.

- [ ] **Step 5: Commit (only if you wired it)**

```bash
git add Backend/IntentClassifier.py reports/latency_local.csv
git commit -m "feat(intent): optional ONNX+INT8 inference path"
```

---

## Task 15: Final sanity pass before viva

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: 48 tests pass, slow tests skipped.

- [ ] **Step 2: Run slow integration tests once (requires HF model)**

Run: `.venv/bin/pytest tests/test_intent_classifier_integration.py -v -m slow`
Expected: 10 tests pass.

- [ ] **Step 3: Run the 30-query smoke test in both modes**

Run: `.venv/bin/python -m scripts.smoke_test`
Expected: ≤ 5 disagreements between local and Groq paths on the demo queries. If more, debug before viva.

- [ ] **Step 4: Run analyze_log to refresh report charts**

Run: `.venv/bin/python -m scripts.analyze_log`
Expected: 3 files written under `reports/`.

- [ ] **Step 5: Verify `Main.py` still runs (Windows machine)**

On the Windows machine: `.venv\Scripts\python Main.py`. Speak/type one of each demo intent. Verify behavior matches expectation.

- [ ] **Step 6: (No commit — this task is only verification.)**

---

## Acceptance criteria

- All 48 fast tests pass.
- All 10 slow integration tests pass against the published HF model.
- Smoke test shows ≤ 5 disagreements between local and Groq on the 30 demo queries.
- `Main.py` runs unchanged on Windows; `FirstLayerDMM` returns the same shape it did before.
- Test-set accuracy ≥ 92%, macro-F1 ≥ 90% (from notebook output).
- Local CPU p95 latency < 100ms (PyTorch); < 30ms if ONNX+INT8 path wired.
- `reports/` has confusion_matrix.png, per_class_metrics.csv, latency_local.csv, plus charts from `analyze_log.py` once Jarvis has been used to populate the log.
- `USE_LOCAL_CLASSIFIER=false` produces identical behavior to today's master branch.
