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
