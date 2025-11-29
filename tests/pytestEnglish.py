import pytest
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "training")

# Add training folder to import path
sys.path.append(TRAINING_DIR)

from trainBaseline_english import predict_with_rules_english



def test_english_hope_prediction():
    text = ["there is hope for everyone"]
    base, final = predict_with_rules_english(text)
    assert final[0] == 1, "Expected Hope speech prediction"


def test_english_nonhope_negation():
    text = ["there is no hope left anymore"]
    base, final = predict_with_rules_english(text)
    assert final[0] == 0, "Expected Non-Hope due to negation phrase"


def test_english_toxic_positivity():
    text = ["just smile and stop being sad"]
    base, final = predict_with_rules_english(text)
    assert final[0] == 0, "Expected Non-Hope for toxic positivity"


def test_english_empty_string():
    text = [""]
    base, final = predict_with_rules_english(text)
    assert final[0] in [0, 1], "Model should return a valid label"
