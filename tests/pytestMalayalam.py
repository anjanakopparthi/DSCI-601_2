import sys, os

# ---------------------------------------------------------
# Add training/ folder to Python path
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "training")
sys.path.append(TRAINING_DIR)

from trainBaseline_malayalam import predict_with_rules_malayalam


def test_malayalam_hope_prediction():
    text = ["എനിക്ക് ഇപ്പോഴും പ്രതീക്ഷയുണ്ട്"]   # "I still have hope"
    base_preds, final = predict_with_rules_malayalam(text)  # ← FIXED: capture both return values
    assert final[0] == 1, "Expected Hope prediction"


def test_malayalam_negation():
    text = ["എനിക്ക് ഒരു പ്രതീക്ഷയും ഇല്ല"]   # "I have no hope"
    base_preds, final = predict_with_rules_malayalam(text)  # ← FIXED: capture both return values
    assert final[0] == 0, "Expected Non-Hope due to negation"


def test_malayalam_toxic_positivity():
    text = ["സാധാരണ ചിരിച്ചുകൊണ്ടിരിക്ക്"]   # "Just keep smiling"
    base_preds, final = predict_with_rules_malayalam(text)  # ← FIXED: capture both return values
    assert final[0] == 0, "Expected Non-Hope for toxic positivity"


def test_malayalam_script_handling():
    text = ["പ്രതീക്ഷ🙂"]
    base_preds, final = predict_with_rules_malayalam(text)  # ← FIXED: capture both return values
    assert final[0] in [0, 1], "Emoji + script mixing should work"