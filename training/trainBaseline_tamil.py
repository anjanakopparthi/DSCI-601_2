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
# 1. Paths  (Modify ONLY these if needed)
# ============================================================

base_dir = Path.cwd()      # Should be: dsci601/DSCI-601_2
processed_dir = base_dir / "processed"
models_dir = base_dir / "models"     # SAME AS MALAYALAM
models_dir.mkdir(parents=True, exist_ok=True)

train_path = processed_dir / "tamil_hope_first_train_corrected.csv"
dev_path   = processed_dir / "tamil_hope_first_dev_parsed.csv"
test_path  = processed_dir / "tamil_hope_first_test_parsed.csv"

print("Loading data...")
train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path) if os.path.exists(dev_path) else None
test_df  = pd.read_csv(test_path) if os.path.exists(test_path) else None


# ============================================================
# 2. Keep only 0/1 labels (drop "not-Tamil" = 2)
# ============================================================

train_df = train_df[train_df["label"].isin([0, 1])].copy()
if dev_df is not None:
    dev_df = dev_df[dev_df["label"].isin([0, 1])].copy()
if test_df is not None:
    test_df = test_df[test_df["label"].isin([0, 1])].copy()


# ============================================================
# 3. Clean text column
# ============================================================

def clean(df):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    return df

train_df = clean(train_df)
if dev_df is not None:
    dev_df = clean(dev_df)
if test_df is not None:
    test_df = clean(test_df)


# ============================================================
# 4. Balance the TRAIN set (undersample majority class)
# ============================================================

maj = train_df[train_df.label == 0]  # Non_hope
minr = train_df[train_df.label == 1] # Hope

print("\nOriginal train distribution:")
print(train_df["label"].value_counts())

maj_under = resample(
    maj,
    replace=False,
    n_samples=len(minr),
    random_state=42
)

train_balanced = pd.concat([maj_under, minr]).sample(frac=1, random_state=42)

print("\nBalanced train distribution:")
print(train_balanced["label"].value_counts())


# ============================================================
# 5. Assign X/y
# ============================================================

X_train = train_balanced["text"]
y_train = train_balanced["label"]

if dev_df is not None:
    X_dev = dev_df["text"]
    y_dev = dev_df["label"]
else:
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

print("\nDataset sizes:")
print("Train:", len(X_train))
print("Dev:  ", len(X_dev))
print("Test: ", len(test_df) if test_df is not None else 0)


# ============================================================
# 6. TF-IDF + Logistic Regression pipeline
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
# 7. Train model
# ============================================================

print("\nTraining model...")
model.fit(X_train, y_train)
print("✓ Model trained!")


# ============================================================
# 8. Evaluate on DEV
# ============================================================

print("\nValidation Results:")
y_dev_pred = model.predict(X_dev)
print(classification_report(y_dev, y_dev_pred, digits=3))
print("Validation Accuracy:", accuracy_score(y_dev, y_dev_pred))


# ============================================================
# 9. Evaluate on TEST
# ============================================================

if test_df is not None:
    X_test = test_df["text"]
    y_test = test_df["label"]

    print("\nTest Results:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=3))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


# ============================================================
# 10. Save the model
# ============================================================

model_path = models_dir / "hope_tamil_model.pkl"
joblib.dump(model, model_path)
print(f"\n✓ Model saved to: {model_path}")


# ============================================================
# 11. Rule-based override definitions
# ============================================================

NEG_PATTERNS = [
    "நம்பிக்கை இல்லை",
    "நம்பிக்கையில்லை",
    "ஆசை இல்லை",
    "முடியாது",
    "நஷ்டம்", 
    "தோல்வி"
]

POS_PATTERNS = [
    "நம்பிக்கை இருக்கிறது",
    "நேற்று விட இன்று நல்லது",
    "வெற்றி",
    "நல்ல நாள்கள் வரும்",
    "நம்பிக்கை வையுங்கள்"
]

def contains_any(text, patterns):
    t = text.lower()
    return any(p in t for p in patterns)

def predict_with_rules_tamil(texts):
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


print("\n============================================")
print("✓ Tamil training complete")
print("Model stored in: dsci601/DSCI-601_2/models/")
print("============================================")
