"""Run 30 demo queries through FirstLayerDMM in both kill-switch modes and
print results side-by-side. Intended as the morning-of-viva sanity check."""
from __future__ import annotations
import os
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
