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
