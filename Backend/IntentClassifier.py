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
