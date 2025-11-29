import sys
import os
import joblib

# ======================= Path Fix ==========================

# Base directory = DSCI-601_2/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# training folder path (for importing functions)
TRAINING_DIR = os.path.join(BASE_DIR, "training")
sys.path.append(TRAINING_DIR)

# model folder path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hope_malayalam_model.pkl")

print("Loading model from:", MODEL_PATH)

# ======================= Load Model =========================

model = joblib.load(MODEL_PATH)

# Import rule-based prediction function
from trainBaseline_malayalam import predict_with_rules_malayalam

# ======================= Test Inputs =========================

texts = [
    "ആശ ഇല്ല",           # negative (should be class 0)
    "ഞങ്ങൾ വിജയിക്കും",    # positive (should be class 1)
    "നല്ല ദിവസം വരും",      # hopeful phrase
    "ഇത് സാധ്യമല്ല"         # negative
]

baseline, final = predict_with_rules_malayalam(texts)

# ======================= Print Results =========================

print("\nBaseline predictions:", baseline)
print("Rule-adjusted predictions:", final)
