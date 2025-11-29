import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "training")

sys.path.append(TRAINING_DIR)


from trainBaseline_english import predict_with_rules_english

texts = ["there is hope", "no hope left", "never lose hope", "not hope", "Just smile and stop being weak"]
baseline, pred = predict_with_rules_english(texts)
print(pred)
print(baseline)
