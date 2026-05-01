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
