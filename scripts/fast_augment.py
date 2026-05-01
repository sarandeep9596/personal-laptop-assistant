"""Generate data/intents.csv and data/test.csv from data/seeds.csv WITHOUT Groq.

Uses casing, keyboard-typo (nlpaug), and simple template rewrites only.
Runs in ~2-3 minutes on any machine. Target: ~200 samples/intent (2000 total).

Usage:
    pip install nlpaug pandas
    python scripts/fast_augment.py
"""
from __future__ import annotations
import random
from pathlib import Path
import pandas as pd

try:
    import nlpaug.augmenter.char as nac
    HAS_NLPAUG = True
except ImportError:
    HAS_NLPAUG = False
    print("[warn] nlpaug not installed — keyboard-typo augmentation skipped. pip install nlpaug")

SEED = 42
N_TEST_PER_INTENT = 15
TARGET_PER_INTENT = 200

ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = ROOT / "data" / "seeds.csv"
TEST_PATH = ROOT / "data" / "test.csv"
INTENTS_PATH = ROOT / "data" / "intents.csv"

# Simple rule-based rewrites per intent — adds lexical variety without an LLM
_REWRITES: dict[str, list[str]] = {
    "open":           ["start {arg}", "launch {arg}", "fire up {arg}", "run {arg}", "boot {arg}"],
    "close":          ["shut down {arg}", "exit {arg}", "kill {arg}", "quit {arg}", "terminate {arg}"],
    "play":           ["put on {arg}", "start {arg}", "stream {arg}", "i want to hear {arg}", "queue {arg}"],
    "google search":  ["look up {arg}", "search {arg}", "find {arg} on google", "google about {arg}", "can you search {arg}"],
    "youtube search": ["look up {arg} on youtube", "find {arg} on youtube", "show me {arg} on youtube", "youtube {arg} please", "search youtube for {arg}"],
    "system":         ["please {arg}", "can you {arg}", "do {arg}", "i need you to {arg}", "{arg} the volume"],
    "realtime":       ["tell me {arg}", "what about {arg}", "can you check {arg}", "find out {arg}", "give me {arg}"],
    "general":        ["hey {arg}", "can you help me with {arg}", "i was wondering {arg}", "explain {arg}", "{arg} please"],
    "reminder":       ["don't forget to {arg}", "ping me for {arg}", "set a reminder {arg}", "alert me {arg}", "notify me {arg}"],
    "generate image": ["create an image of {arg}", "make a picture of {arg}", "draw {arg}", "generate a picture of {arg}", "show me {arg} as an image"],
}


def _extract_arg(query: str, intent: str) -> str:
    """Best-effort: strip the leading intent keyword to get the argument."""
    prefix_map = {
        "open": ("open", "launch", "start"),
        "close": ("close", "shut down", "exit", "kill"),
        "play": ("play",),
        "google search": ("google", "search for", "search"),
        "youtube search": ("youtube", "search on youtube", "search"),
        "system": ("system", "volume", "mute", "increase", "decrease"),
        "realtime": ("what is", "tell me", "realtime", "weather"),
        "general": ("general",),
        "reminder": ("remind me", "reminder", "set a reminder"),
        "generate image": ("generate", "create", "make", "generate a photo of", "generate image"),
    }
    q = query.strip().lower()
    for prefix in prefix_map.get(intent, ()):
        if q.startswith(prefix):
            return q[len(prefix):].strip()
    return query  # fallback: use full query as "arg"


def _rewrite_variants(row: dict) -> list[str]:
    intent, query = row["intent"], row["query"]
    templates = _REWRITES.get(intent, [])
    if not templates:
        return []
    arg = _extract_arg(query, intent)
    variants = []
    for tmpl in templates:
        try:
            v = tmpl.format(arg=arg)
        except KeyError:
            v = query
        if v and v.lower() != query.lower():
            variants.append(v)
    return variants


def holdout_test_set(seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    test_rows, train_rows = [], []
    for _, group in seeds.groupby("intent"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        test_idxs = set(idxs[:N_TEST_PER_INTENT])
        for i in idxs:
            (test_rows if i in test_idxs else train_rows).append(seeds.loc[i])
    return (
        pd.DataFrame(train_rows).reset_index(drop=True),
        pd.DataFrame(test_rows).reset_index(drop=True),
    )


def augment(train_pool: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    keyboard_aug = nac.KeyboardAug(aug_char_min=1, aug_char_max=1, aug_word_p=0.15, include_special_char=False) if HAS_NLPAUG else None

    all_rows: list[dict] = []
    for intent, group in train_pool.groupby("intent"):
        rows = group.to_dict("records")

        # Step 1: template rewrites
        for row in list(rows):
            for v in _rewrite_variants(row):
                rows.append({"query": v, "intent": intent})

        # Step 2: casing + keyboard typo to reach TARGET_PER_INTENT
        base_rows = group.to_dict("records")  # only original seeds as typo source
        while len(rows) < TARGET_PER_INTENT:
            base = rng.choice(base_rows)
            q = base["query"]
            r = rng.random()
            if r < 0.3:
                q_new = q.upper()
            elif r < 0.55:
                q_new = q.capitalize()
            elif r < 0.75 and keyboard_aug:
                try:
                    q_new = keyboard_aug.augment(q)[0]
                except Exception:
                    q_new = q
            else:
                q_new = q.lower()
            rows.append({"query": q_new, "intent": intent})

        all_rows.extend(rows[:TARGET_PER_INTENT])

    return pd.DataFrame(all_rows)


def main():
    seeds = pd.read_csv(SEEDS_PATH)
    print(f"Loaded {len(seeds)} seeds ({seeds['intent'].nunique()} intents)")

    train_pool, test = holdout_test_set(seeds)
    test.to_csv(TEST_PATH, index=False)
    print(f"test.csv: {len(test)} rows")
    print(f"train pool (seeds only): {len(train_pool)} rows")

    final = augment(train_pool)
    final = final.sample(frac=1, random_state=SEED).reset_index(drop=True)
    final.to_csv(INTENTS_PATH, index=False)
    print(f"intents.csv: {len(final)} rows")
    print(final["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
