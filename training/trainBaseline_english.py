import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ============================================================
# 1. Paths
# ============================================================

base_dir = Path.cwd()
processed_dir = base_dir / "processed"
models_dir = base_dir / "models"
models_dir.mkdir(exist_ok=True)

train_path = processed_dir / "english_hope_train_balanced_undersample.csv"
dev_path   = processed_dir / "english_hope_dev_parsed.csv"
test_path  = processed_dir / "english_hope_test_parsed.csv"

print(f"Loading data:")
print(f"  Train: {train_path}")
print(f"  Dev:   {dev_path}")
print(f"  Test:  {test_path}\n")

# ============================================================
# 2. Load data
# ============================================================

train_df = pd.read_csv(train_path)
print(f"Train loaded: {len(train_df)} rows")

dev_df = pd.read_csv(dev_path) if dev_path.exists() else None
test_df = pd.read_csv(test_path) if test_path.exists() else None

if dev_df is not None:
    print(f"Dev loaded: {len(dev_df)} rows")
if test_df is not None:
    print(f"Test loaded: {len(test_df)} rows")

# ============================================================
# 3. Keep ONLY labels 0/1 and clean text
# ============================================================

def clean_df(df):
    df = df[df["label"].isin([0, 1])].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    return df

train_df = clean_df(train_df)
if dev_df is not None:
    dev_df = clean_df(dev_df)
if test_df is not None:
    test_df = clean_df(test_df)

print(f"Train after cleaning: {len(train_df)} rows")
if dev_df is not None:
    print(f"Dev after cleaning:   {len(dev_df)} rows")
if test_df is not None:
    print(f"Test after cleaning:  {len(test_df)} rows")

# ============================================================
# 4. Validation split logic (fixed)
# ============================================================

X_train = train_df["text"]
y_train = train_df["label"]

if dev_df is not None and len(dev_df) > 5:
    print("\nUsing provided dev set.")
    X_dev = dev_df["text"]
    y_dev = dev_df["label"]
else:
    print("\nDev set too small → Using split from train")
    try:
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
    except ValueError:
        print("⚠ Stratified split failed → using non-stratified split")
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42
        )

print("\nFinal dataset sizes:")
print(f"Train: {len(X_train)}")
print(f"Dev:   {len(X_dev)}\n")

# ============================================================
# 6. Model pipeline
# ============================================================

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3)
    )),
    ('clf', LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# ============================================================
# 7. Train
# ============================================================

print("Training model...")
model.fit(X_train, y_train)
print("✓ Model trained!")

# ============================================================
# 8. Validation
# ============================================================

if len(X_dev) > 0:
    print("\nValidation Results:")
    y_pred = model.predict(X_dev)
    print(classification_report(y_dev, y_pred, digits=3))
    print("Accuracy:", accuracy_score(y_dev, y_pred))
else:
    print("\n⚠ No validation data")

# ============================================================
# 9. Test
# ============================================================

if test_df is not None and len(test_df) > 0:
    print("\nTest Results:")
    X_test = test_df["text"]
    y_test = test_df["label"]
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=3))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ============================================================
# 10. Save model
# ============================================================

model_path = models_dir / "hope_english_model.pkl"
joblib.dump(model, model_path)
print(f"\n✓ Saved model at {model_path}")

# ============================================================
# 11. Reload model
# ============================================================

model = joblib.load(model_path)
print("✓ Reloaded model")

# ============================================================
# 12. Rule-based wrapper
# ============================================================

NEG_PATTERNS = [
    "no hope", "not much hope", "little hope", "without hope",
    "hopeless", "don't have hope", "do not have hope", "not hope"
]

POS_PATTERNS = [
    "there is hope", "i have hope", "we have hope", "never lose hope",
    "don't lose hope", "do not lose hope", "never give up hope",
    "keep hope alive", "stay hopeful", "things will be good",
    "better days are coming", "have hope"
]

def contains_any(text, patterns):
    text = text.lower()
    return any(p in text for p in patterns)

def predict_with_rules_english(texts):
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

print("\n" + "="*60)
print("✓ Training complete (fixed + safe version)")
print("="*60)
