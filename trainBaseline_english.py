import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ============================================================
# 1. Paths to your ENGLISH processed CSVs
#    Adjust these paths to match your local files
# ============================================================

# Choose ONE train file: balanced undersample, oversample, or original parsed.
# A) Balanced (undersampled) train
train_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\english_hope_train_balanced_undersample.csv"

# B) Balanced (oversampled) train
# train_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\english_hope_train_balanced_oversample.csv"

# C) Original parsed (imbalanced) train
# train_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\english_hope_train_parsed.csv"

dev_path  = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\english_hope_dev_parsed.csv"
test_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\english_hope_test_parsed.csv"

# ============================================================
# 2. Load data
#    Files should have: text, label_str, label
#    label: 0 = Non_hope_speech, 1 = Hope_speech, 2 = not-English
# ============================================================

train_df = pd.read_csv(train_path)

dev_df = pd.read_csv(dev_path) if os.path.exists(dev_path) else None
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# ============================================================
# 3. Keep ONLY English hope vs non-hope (drop not-English)
# ============================================================

train_df = train_df[train_df["label"].isin([0, 1])].copy()

if dev_df is not None:
    dev_df = dev_df[dev_df["label"].isin([0, 1])].copy()

if test_df is not None:
    test_df = test_df[test_df["label"].isin([0, 1])].copy()

# ============================================================
# 4. Handle missing values
# ============================================================

train_df["text"] = train_df["text"].fillna("")

if dev_df is not None:
    dev_df["text"] = dev_df["text"].fillna("")

if test_df is not None:
    test_df["text"] = test_df["text"].fillna("")

# ============================================================
# 5. Train–validation split
#    If dev set is available, use it as validation.
#    Otherwise, split from train.
# ============================================================

X_train = train_df["text"]
y_train = train_df["label"]

if dev_df is not None:
    X_dev = dev_df["text"]
    y_dev = dev_df["label"]
else:
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

print("Training class distribution:")
print(y_train.value_counts())
print(f"\nClass 0 (Non_hope_speech): {(y_train == 0).sum()} samples")
print(f"Class 1 (Hope_speech): {(y_train == 1).sum()} samples")

# ============================================================
# 6. TF-IDF + Logistic Regression pipeline
# ============================================================

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3)   # unigrams + bigrams + trigrams
    )),
    ('clf', LogisticRegression(
        max_iter=500,
        class_weight='balanced',  # still useful even if balanced
        n_jobs=-1
    ))
])

# ============================================================
# 7. Train
# ============================================================

print("\nTraining model...")
model.fit(X_train, y_train)

# ============================================================
# 8. Evaluate on validation
# ============================================================

print("\nValidation Results:")
y_pred = model.predict(X_dev)
print(classification_report(y_dev, y_pred, digits=3))
print("Validation Accuracy:", accuracy_score(y_dev, y_pred))

# ============================================================
# 9. Evaluate on test set if available
# ============================================================

if test_df is not None:
    X_test = test_df["text"]
    y_test = test_df["label"]

    print("\nTest Results:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=3))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ============================================================
# 10. Save model
# ============================================================

model_path = "hope_english_model.pkl"
joblib.dump(model, model_path)
print(f"\n Model saved as {model_path}")

# ============================================================
# 11. Load model again (like you do in testing scripts)
# ============================================================

model = joblib.load(model_path)
print("\n Model reloaded for inference")

# ============================================================
# 12. Wrapper with rules around the model
#     - Negative hope expressions → force label 0
#     - Strong positive hope expressions → force label 1
# ============================================================

# Negative-hope patterns → should be class 0 (Non_hope_speech)
NEG_PATTERNS = [
    "no hope",
    "not much hope",
    "little hope",
    "without hope",
    "hopeless",
    "don't have hope",
    "do not have hope",
    "not hope",          # explicitly negative
]

# Strong positive hope patterns → should be class 1 (Hope_speech)
POS_PATTERNS = [
    "there is hope",
    "i have hope",
    "we have hope",
    "never lose hope",   # explicitly positive
    "don't lose hope",
    "do not lose hope",
    "never give up hope",
    "keep hope alive",
    "stay hopeful",
    "things will be good",
    "better days are coming",
    "have hope",
]

def contains_any_substring(text: str, patterns):
    t = text.lower()
    return any(p in t for p in patterns)

def predict_with_rules_english(texts):
    """
    texts: list[str]
    returns: numpy array of final labels (0/1)
    """
    base_preds = model.predict(texts)
    fixed = base_preds.copy()

    for i, txt in enumerate(texts):
        low = txt.lower()

        # 1) Explicit positive-hope phrases → force Hope_speech (1)
        if contains_any_substring(low, POS_PATTERNS):
            fixed[i] = 1
            print(f" Positive override → Hope_speech (1): {txt}")
            continue

        # 2) Explicit negative-hope phrases → force Non_hope_speech (0)
        if contains_any_substring(low, NEG_PATTERNS):
            fixed[i] = 0
            print(f" Negative override → Non_hope_speech (0): {txt}")
            continue

    return fixed

