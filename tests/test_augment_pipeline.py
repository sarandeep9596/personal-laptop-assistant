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
