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
