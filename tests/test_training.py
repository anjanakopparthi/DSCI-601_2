"""
Tests for the baseline training module (training/train_all.py) and the
saved model artifacts.

Covers:
    - build_model constructs a valid TF-IDF + LogReg pipeline
    - predict_with_rules override logic (positive precedence, case-insensitive,
      no-op when no pattern matches, works on Tamil/Malayalam scripts + emoji)
    - compute_metrics returns the expected keys and sane values
    - saved base models predict on all three languages
    - saved rules bundles load via load_rules_model (the old closure-pickle
      bug regression test)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH / "training"))

from train_all import (build_model, compute_metrics, load_rules_model,
                       predict_with_rules)


# ============================================================
# Unit tests: model building
# ============================================================

class TestBuildModel:

    def test_pipeline_structure(self):
        model = build_model(
            {"max_features": 100, "ngram_range": [1, 2]},
            {"max_iter": 100},
        )
        assert model.steps[0][0] == "tfidf"
        assert model.steps[1][0] == "clf"

    def test_ngram_list_converted_to_tuple(self):
        model = build_model(
            {"max_features": 100, "ngram_range": [1, 3]},
            {"max_iter": 100},
        )
        assert model.named_steps["tfidf"].ngram_range == (1, 3)

    def test_fits_multilingual_text(self):
        model = build_model({"max_features": 50}, {"max_iter": 100})
        X = pd.Series(["நம்பிக்கை இருக்கிறது", "പ്രതീക്ഷയുണ്ട്",
                       "there is hope", "இல்லை", "ഇല്ല", "no hope"] * 5)
        y = pd.Series([1, 1, 1, 0, 0, 0] * 5)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds) <= {0, 1}


# ============================================================
# Unit tests: rule-based overrides
# ============================================================

class _ConstantModel:
    """Mock model that always predicts the same class."""

    def __init__(self, value):
        self.value = value

    def predict(self, texts):
        return np.full(len(texts), self.value, dtype=int)


class TestPredictWithRules:

    def test_positive_pattern_forces_hope(self):
        preds = predict_with_rules(_ConstantModel(0),
                                   ["never lose hope friends"],
                                   ["never lose hope"], ["no hope"])
        assert preds[0] == 1

    def test_negative_pattern_forces_non_hope(self):
        preds = predict_with_rules(_ConstantModel(1),
                                   ["this is hopeless"],
                                   ["never lose hope"], ["hopeless"])
        assert preds[0] == 0

    def test_positive_takes_precedence(self):
        # text matches both a positive and a negative pattern
        preds = predict_with_rules(_ConstantModel(0),
                                   ["never lose hope even when hopeless"],
                                   ["never lose hope"], ["hopeless"])
        assert preds[0] == 1

    def test_no_match_keeps_model_prediction(self):
        preds = predict_with_rules(_ConstantModel(1),
                                   ["completely unrelated text"],
                                   ["never lose hope"], ["hopeless"])
        assert preds[0] == 1

    def test_case_insensitive(self):
        preds = predict_with_rules(_ConstantModel(0),
                                   ["NEVER LOSE HOPE!!!"],
                                   ["never lose hope"], [])
        assert preds[0] == 1

    def test_tamil_patterns(self):
        preds = predict_with_rules(_ConstantModel(0),
                                   ["நம்பிக்கை இருக்கிறது நண்பா"],
                                   ["நம்பிக்கை இருக்கிறது"], ["நம்பிக்கை இல்லை"])
        assert preds[0] == 1

    def test_malayalam_patterns_with_emoji(self):
        preds = predict_with_rules(_ConstantModel(0),
                                   ["പ്രതീക്ഷയുണ്ട് 🙂"],
                                   ["പ്രതീക്ഷയുണ്ട്"], [])
        assert preds[0] == 1

    def test_empty_patterns_never_override(self):
        preds = predict_with_rules(_ConstantModel(1),
                                   ["hope hopeless whatever"], [], [])
        assert preds[0] == 1


# ============================================================
# Unit tests: metrics
# ============================================================

class TestComputeMetrics:

    def test_keys_and_perfect_score(self):
        m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1])
        for key in ("accuracy", "macro_f1", "weighted_f1", "hope_f1",
                    "hope_precision", "hope_recall", "confusion_matrix"):
            assert key in m
        assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0

    def test_confusion_matrix_layout(self):
        # y_true: [0,0,1,1], y_pred: [0,1,1,1] -> TN=1 FP=1 FN=0 TP=2
        m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        assert m["confusion_matrix"] == [[1, 1], [0, 2]]


# ============================================================
# Saved model artifact tests (skip if models not trained yet)
# ============================================================

LANGS = ["english", "tamil", "malayalam"]


@pytest.mark.parametrize("lang", LANGS)
def test_base_model_predicts(lang, config_data, base_dir):
    import joblib
    path = base_dir / config_data.get("models_dir", "models") / \
        f"hope_{lang}_model.pkl"
    if not path.exists():
        pytest.skip(f"{path} not found — run training/train_all.py")
    model = joblib.load(path)
    preds = model.predict(["some text", "வணக்கம்", "നമസ്കാരം"])
    assert set(preds) <= {0, 1}


@pytest.mark.parametrize("lang", LANGS)
def test_rules_bundle_loads_and_predicts(lang, config_data, base_dir):
    """Regression test for the old closure-pickle corruption bug."""
    path = base_dir / config_data.get("models_dir", "models") / \
        f"hope_{lang}_model_with_rules.pkl"
    if not path.exists():
        pytest.skip(f"{path} not found — run training/train_all.py")
    model, pos, neg = load_rules_model(path)
    assert isinstance(pos, list) and isinstance(neg, list)
    preds = predict_with_rules(model, ["hello world"], pos, neg)
    assert preds[0] in (0, 1)
