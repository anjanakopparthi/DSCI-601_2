import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ============================================================
# 1. Correct paths
# ============================================================

base_dir = Path.cwd()   # should be: dsci601/DSCI-601_2
processed_dir = base_dir / "processed"
models_dir = base_dir / "models"       # <-- YOUR REQUIRED PATH
models_dir.mkdir(parents=True, exist_ok=True)

train_path = processed_dir / "malayalam_hope_train_processed.csv"
dev_path   = processed_dir / "malayalam_hope_dev_processed.csv"
test_path  = processed_dir / "malayalam_hope_test_processed.csv"

print("Loading data...")
train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)
test_df  = pd.read_csv(test_path)

# ============================================================
# 2. Clean text + ensure labels are valid (0 or 1)
# ============================================================

def clean(df):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    df = df[df["label"].isin([0, 1])]
    return df

train_df = clean(train_df)
dev_df   = clean(dev_df)
test_df  = clean(test_df)

print("\nClass counts:")
print("Train:\n", train_df["label"].value_counts())
print("Dev:\n", dev_df["label"].value_counts())
print("Test:\n", test_df["label"].value_counts())

# ============================================================
# 3. Balance TRAIN dataset
# ============================================================

maj = train_df[train_df.label == 0]
minr = train_df[train_df.label == 1]

print(f"\nOriginal Train → Non_hope={len(maj)}, Hope={len(minr)}")

n_min = len(minr)
maj_under = resample(maj, replace=False, n_samples=n_min, random_state=42)

train_balanced = pd.concat([maj_under, minr]).sample(frac=1, random_state=42)

print("\nBalanced Train:\n", train_balanced["label"].value_counts())

# ============================================================
# 4. Assign X/y
# ============================================================

X_train = train_balanced["text"]
y_train = train_balanced["label"]

X_dev = dev_df["text"]
y_dev = dev_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

print("\nFinal Sizes:")
print("Train:", len(X_train))
print("Dev:  ", len(X_dev))
print("Test: ", len(X_test))

# ============================================================
# 5. Create TF-IDF + Logistic Regression pipeline
# ============================================================

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1))
])

# ============================================================
# 6. Train
# ============================================================

print("\nTraining model...")
model.fit(X_train, y_train)
print("✓ Model trained!")

# ============================================================
# 7. Validation results
# ============================================================

print("\nValidation Results:")
y_dev_pred = model.predict(X_dev)
print(classification_report(y_dev, y_dev_pred, digits=3))
print("Validation Accuracy:", accuracy_score(y_dev, y_dev_pred))

# ============================================================
# 8. Test results
# ============================================================

print("\nTest Results:")
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred, digits=3))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ============================================================
# 9. Save model (YOUR PATH)
# ============================================================

model_path = models_dir / "hope_malayalam_model.pkl"
joblib.dump(model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# ============================================================
# 10. Rule-based override
# ============================================================

NEG_PATTERNS = [
    "ഇല്ല", "നഷ്ടപ്പെട്ടു", "കിടയില്ല",
    "മാർഗ്ഗമില്ല", "മുട്ടി", "സാധ്യമല്ല"
]

POS_PATTERNS = [
    "ആശയുണ്ട്", "വിജയിക്കും", "വിജയം",
    "നല്ല ദിവസം", "വിശ്വസിക്കുന്നു"
]

def contains_any(text, patterns):
    t = text.lower()
    return any(p in t for p in patterns)

def predict_with_rules_malayalam(texts):
    base_preds = model.predict(texts)
    fixed = base_preds.copy()

    for i, txt in enumerate(texts):
        if contains_any(txt, POS_PATTERNS):
            fixed[i] = 1
            continue
        if contains_any(txt, NEG_PATTERNS):
            fixed[i] = 0
            continue

    return base_preds, fixed

print("\n========================================")
print("✓ Malayalam training complete")
print("Model stored in: dsci601/DSCI-601_2/models/")
print("========================================")
