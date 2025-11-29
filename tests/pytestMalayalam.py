def test_malayalam_hope_prediction(malayalam_model):
    text = "എനിക്ക് ഇപ്പോഴും പ്രതീക്ഷയുണ്ട്"
    pred = malayalam_model.predict([text])[0]
    assert pred == 1, "Expected Hope prediction"

def test_malayalam_negation(malayalam_model):
    text = "എനിക്ക് ഒരു പ്രതീക്ഷയും ഇല്ല"
    pred = malayalam_model.predict([text])[0]
    assert pred == 0, "Expected Non-Hope due to negation"

def test_malayalam_toxic_positivity(malayalam_model):
    text = "സാധാരണ ചിരിച്ചുകൊണ്ടിരിക്ക്"
    pred = malayalam_model.predict([text])[0]
    assert pred == 0, "Expected Non-Hope for toxic positivity"

def test_malayalam_script_handling(malayalam_model):
    text = "പ്രതീക്ഷ🙂"
    pred = malayalam_model.predict([text])[0]
    assert pred in [0,1], "Emoji + script mixing should work"
