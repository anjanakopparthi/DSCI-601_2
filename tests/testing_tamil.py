import sys
import os
import joblib

# =============== FIX PATHS ==================

# Base directory: DSCI-601_2/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# training folder path
TRAINING_DIR = os.path.join(BASE_DIR, "training")
sys.path.append(TRAINING_DIR)

# model folder path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hope_tamil_model.pkl")

print("Loading model from:", MODEL_PATH)

# ============================================
# Load model
# ============================================

model = joblib.load(MODEL_PATH)

# Import override function from training file
from trainBaseline_tamil import predict_with_rules_tamil

# ============================================
# Test sentences
# ============================================

texts = [
    "எனக்கு நம்பிக்கை இருக்கிறது",
    "நம்பிக்கை இல்லை",
    "இது நல்ல நாள் வரும்",
    "நம்மால் முடியாது",
    "உற்சாகமாக இரு நண்பா"
]

baseline, final = predict_with_rules_tamil(texts)

# ============================================
# Print Results
# ============================================

print("\nBase model predictions:", baseline)
print("Rule-adjusted predictions:", final)
