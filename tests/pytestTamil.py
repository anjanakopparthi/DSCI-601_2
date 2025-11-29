import sys, os

# ---------------------------------------------------------
# Add training/ folder to Python path
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "training")
sys.path.append(TRAINING_DIR)

from trainBaseline_tamil import predict_with_rules_tamil


def test_tamil_hope_prediction():
    text = ["நம்பிக்கை இருக்கிறது"]   # "There is hope"
    base, final = predict_with_rules_tamil(text)
    assert final[0] == 1, "Expected Hope speech prediction"


def test_tamil_negation():
    text = ["என்னிடம் நம்பிக்கை எதுவும் இல்லை"]   # "I have no hope"
    base, final = predict_with_rules_tamil(text)
    assert final[0] == 0, "Expected Non-Hope due to negation phrase"


def test_tamil_toxic_positivity():
    text = ["சிரித்து விடு, அது எளிது"]   # "Just smile, it's easy"
    base, final = predict_with_rules_tamil(text)
    assert final[0] == 0, "Expected Non-Hope for toxic positivity"


def test_tamil_unicode_handling():
    text = ["நம்பிக்கை💛"]
    base, final = predict_with_rules_tamil(text)
    assert final[0] in [0, 1], "Unicode emojis should not break model"
