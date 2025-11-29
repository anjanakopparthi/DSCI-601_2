def test_english_hope_prediction(english_model):
    text = "there is hope for everyone"
    pred = english_model.predict([text])[0]
    assert pred == 1, "Expected Hope speech prediction"

def test_english_nonhope_negation(english_model):
    text = "there is no hope left anymore"
    pred = english_model.predict([text])[0]
    assert pred == 0, "Expected Non-Hope due to negation"

def test_english_toxic_positivity(english_model):
    text = "just smile and stop being sad"
    pred = english_model.predict([text])[0]
    assert pred == 0, "Expected Non-Hope for toxic positivity"

def test_english_empty_string(english_model):
    text = ""
    pred = english_model.predict([text])[0]
    assert pred in [0,1], "Model should return a valid label"
