import joblib
model = joblib.load("hope_tamil_model.pkl")
text = ["இது ஒரு நல்ல நாள்", "நம்பிக்கை இல்லை"]
pred = model.predict(text)
print(pred)
