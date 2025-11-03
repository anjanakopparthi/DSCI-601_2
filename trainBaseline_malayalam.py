import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ============================================================
# 1. Paths to your MALAYALAM original CSVs
#    (these are the ones you uploaded: malayalam_train/dev/test.csv)
# ============================================================

train_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\malayalam_train.csv"
dev_path   = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\malayalam_dev.csv"
test_path  = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\malayalam_test.csv"

# ============================================================
# 2. Load raw data
#    Expecting columns: 'text', 'label' (but we'll REBUILD labels)
# ============================================================

train_df_raw = pd.read_csv(train_path)
dev_df_raw   = pd.read_csv(dev_path)
test_df_raw  = pd.read_csv(test_path)

# ============================================================
# 3. Derive label_str from the text itself
#    The text contains suffixes like:
#      - 'Non_hope_speech'
#      - 'Hope_speech'
#      - 'notmalayalam'
# ============================================================

def derive_label_str(text: str) -> str:
    s = str(text)
    if "notmalayalam" in s:
        return "not-Malayalam"
    if "Non_hope_speech" in s:
        return "Non_hope_speech"
    if "Hope_speech" in s:
        return "Hope_speech"
    return None  # rare edge case; we can drop later if needed

for df in (train_df_raw, dev_df_raw, test_df_raw):
    df["label_str"] = df["text"].apply(derive_label_str)

label_map = {
    "Non_hope_speech": 0,
    "Hope_speech": 1,
    "not-Malayalam": 2,
}

for df in (train_df_raw, dev_df_raw, test_df_raw):
    df["label"] = df["label_str"].map(label_map)

# Optional: print counts to verify
print("Train label_str counts:\n", train_df_raw["label_str"].value_counts(dropna=False))
print("\nDev label_str counts:\n", dev_df_raw["label_str"].value_counts(dropna=False))
print("\nTest label_str counts:\n", test_df_raw["label_str"].value_counts(dropna=False))

# ============================================================
# 4. Prepare final parsed DataFrames, and filter out not-Malayalam
#    We keep only label 0 and 1 for the HOPE classifier.
# ============================================================

train_parsed = train_df_raw[["text", "label_str", "label"]].copy()
dev_parsed   = dev_df_raw[["text", "label_str", "label"]].copy()
test_parsed  = test_df_raw[["text", "label_str", "label"]].copy()

train_bin = train_parsed[train_parsed["label"].isin([0, 1])].copy()
dev_bin   = dev_parsed[dev_parsed["label"].isin([0, 1])].copy()
test_bin  = test_parsed[test_parsed["label"].isin([0, 1])].copy()

print("\nFiltered to Malayalam only (labels 0/1):")
print("Train:", train_bin["label"].value_counts())
print("Dev:", dev_bin["label"].value_counts())
print("Test:", test_bin["label"].value_counts())

# ============================================================
# 5. Balance the TRAIN set (same idea as English)
#    Majority = Non_hope_speech (0), Minority = Hope_speech (1)
# ============================================================

maj = train_bin[train_bin["label"] == 0]  # Non_hope_speech
minr = train_bin[train_bin["label"] == 1] # Hope_speech

n_maj = len(maj)
n_min = len(minr)

print(f"\nOriginal train counts -> Non_hope_speech (0): {n_maj}, Hope_speech (1): {n_min}")

# A) UNDERSAMPLE majority to minority size
maj_under = resample(
    maj,
    replace=False,
    n_samples=n_min,
    random_state=42
)
train_balanced_under = pd.concat([maj_under, minr], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

# B) OVERSAMPLE minority to majority size (if you want to try later)
min_over = resample(
    minr,
    replace=True,
    n_samples=n_maj,
    random_state=42
)
train_balanced_over = pd.concat([maj, min_over], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

print("\nBalanced (undersample) train counts:")
print(train_balanced_under["label"].value_counts())

print("\nBalanced (oversample) train counts:")
print(train_balanced_over["label"].value_counts())

# Choose which balanced set to use for training:
use_undersample = True

if use_undersample:
    train_final = train_balanced_under
    print("\n➡ Using UNDERSAMPLED balanced train set for model training.")
else:
    train_final = train_balanced_over
    print("\n➡ Using OVERSAMPLED balanced train set for model training.")

# ============================================================
# 6. Handle missing values
# ============================================================

for df in (train_final, dev_bin, test_bin):
    df["text"] = df["text"].fillna("")

X_train = train_final["text"]
y_train = train_final["label"]

X_dev = dev_bin["text"]
y_dev = dev_bin["label"]

X_test = test_bin["text"]
y_test = test_bin["label"]

# ============================================================
# 7. Build TF-IDF + Logistic Regression pipeline
# ============================================================

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3)  # unigrams + bigrams + trigrams
    )),
    ('clf', LogisticRegression(
        max_iter=500,
        class_weight='balanced',  # okay to keep; helps robustness
        n_jobs=-1
    ))
])

# ============================================================
# 8. Train
# ============================================================

print("\nTraining model...")
model.fit(X_train, y_train)

# ============================================================
# 9. Evaluate on validation (dev) set
# ============================================================

print("\nValidation Results (DEV):")
y_dev_pred = model.predict(X_dev)
print(classification_report(y_dev, y_dev_pred, digits=3))
print("Validation Accuracy:", accuracy_score(y_dev, y_dev_pred))

# ============================================================
# 10. Evaluate on TEST set
# ============================================================

print("\nTest Results:")
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred, digits=3))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ============================================================
# 11. Save model
# ============================================================

model_path = "hope_malayalam_model.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Malayalam model saved as {model_path}")

# ============================================================
# 12. (Optional) Reload model for inference
# ============================================================

model = joblib.load(model_path)
print("\n✅ Malayalam model reloaded for inference")

# ============================================================
# 13. Rule-based wrapper for Malayalam predictions
#     - Negative hope expressions → force label 0
#     - Strong positive hope expressions → force label 1
# ============================================================

# Negative/“no hope” phrases → Non_hope_speech (0)
NEG_PATTERNS = [
    "ഇല്ല",          # no / not
    "നഷ്ടപ്പെട്ടു",    # lost
    "കിടയില്ല",      # does not exist
    "മാർഗ്ഗമില്ല",    # no way
    "മുട്ടി",         # stuck / finished
    "സാധ്യമല്ല",       # not possible
]

# Positive hope phrases → Hope_speech (1)
POS_PATTERNS = [
    "ആശയുണ്ട്",      # have hope
    "വിജയിക്കും",    # will win
    "വിജയം",        # success
    "നല്ല ദിവസം",    # good day
    "വിശ്വസിക്കുന്നു", # believe
]

def contains_any(text: str, patterns) -> bool:
    t = text.lower()
    return any(p in t for p in patterns)

def predict_with_rules_malayalam(texts):
    """
    texts: list[str]
    Returns: numpy array of final labels (0 = Non_hope_speech, 1 = Hope_speech)
    """
    base_preds = model.predict(texts)
    fixed = base_preds.copy()

    for i, txt in enumerate(texts):
        t = txt.lower()

        # 1) Explicit positive hope → force Hope_speech (1)
        if contains_any(t, POS_PATTERNS):
            fixed[i] = 1
            print(f"✅ Positive override → Hope_speech (1): {txt}")
            continue

        # 2) Explicit negative/“no hope” → force Non_hope_speech (0)
        if contains_any(t, NEG_PATTERNS):
            fixed[i] = 0
            print(f"⚠️ Negative override → Non_hope_speech (0): {txt}")
            continue

    return fixed

