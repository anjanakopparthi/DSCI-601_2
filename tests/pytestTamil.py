def test_tamil_hope_prediction(tamil_model):
    text = "நம்பிக்கை இருக்கிறது"
    pred = tamil_model.predict([text])[0]
    assert pred == 1, "Expected Hope speech prediction"

def test_tamil_negation(tamil_model):
    text = "என்னிடம் நம்பிக்கை எதுவும் இல்லை"
    pred = tamil_model.predict([text])[0]
    assert pred == 0, "Expected Non-Hope due to negation"

def test_tamil_toxic_positivity(tamil_model):
    text = "சிரித்து விடு, அது எளிது"
    pred = tamil_model.predict([text])[0]
    assert pred == 0, "Expected Non-Hope for toxic positivity"

def test_tamil_unicode_handling(tamil_model):
    text = "நம்பிக்கை💛"
    pred = tamil_model.predict([text])[0]
    assert pred in [0,1], "Unicode emojis should not break model"
