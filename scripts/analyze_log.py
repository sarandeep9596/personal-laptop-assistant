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
