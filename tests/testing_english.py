from trainBaseline_english import predict_with_rules_english

texts = ["there is hope", "no hope left", "never lose hope", "not hope"]
baseline, pred = predict_with_rules_english(texts)
print(pred)
print(baseline)
