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
