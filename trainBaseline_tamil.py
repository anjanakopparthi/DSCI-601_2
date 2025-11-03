import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# ==========
# 1. Paths to your (NEW) processed CSVs
#    Use the corrected / parsed files you downloaded from ChatGPT
# ==========
train_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\tamil_hope_first_train_corrected.csv"
dev_path   = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\tamil_hope_first_dev_parsed.csv"
test_path  = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\tamil_hope_first_test_parsed.csv"

# ==========
# 2. Load data
#    These files already have columns: text, label_str, label
# ==========
train_df = pd.read_csv(train_path)

dev_df = pd.read_csv(dev_path) if os.path.exists(dev_path) else None
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# ==========
# 3. Keep ONLY Tamil hope vs non-hope
#    label: 0 = Non_hope_speech, 1 = Hope_speech, 2 = not-Tamil
# ==========
train_df = train_df[train_df["label"].isin([0, 1])].copy()

if dev_df is not None:
    dev_df = dev_df[dev_df["label"].isin([0, 1])].copy()

if test_df is not None:
    test_df = test_df[test_df["label"].isin([0, 1])].copy()

# ==========
# 4. Handle missing values
# ==========
train_df["text"] = train_df["text"].fillna("")

if dev_df is not None:
    dev_df["text"] = dev_df["text"].fillna("")

if test_df is not None:
    test_df["text"] = test_df["text"].fillna("")

# ==========
# 5. Train–validation split
#    If dev set is available, use it as validation.
#    Otherwise, split from train.
# ==========
X_train = train_df["text"]
y_train = train_df["label"]

if dev_df is not None:
    X_dev = dev_df["text"]
    y_dev = dev_df["label"]
else:
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

# Print class distribution
print("Training class distribution:")
print(y_train.value_counts())
print(f"\nClass 0 (Non_hope_speech): {(y_train == 0).sum()} samples")
print(f"Class 1 (Hope_speech): {(y_train == 1).sum()} samples")

# ==========
# 6. TF-IDF + Logistic Regression pipeline
# ==========
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3)  # unigrams + bigrams + trigrams
    )),
    ('clf', LogisticRegression(
        max_iter=500,
        class_weight='balanced',  # handle any imbalance
        n_jobs=-1
    ))
])

# ==========
# 7. Train
# ==========
print("\nTraining model...")
model.fit(X_train, y_train)

# ==========
# 8. Evaluate on validation
# ==========
print("\nValidation Results:")
y_pred = model.predict(X_dev)
print(classification_report(y_dev, y_pred, digits=3))
print("Validation Accuracy:", accuracy_score(y_dev, y_pred))

# ==========
# 9. Evaluate on test set if available
# ==========
if test_df is not None:
    X_test = test_df["text"]
    y_test = test_df["label"]

    print("\nTest Results:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=3))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ==========
# 10. Save model
# ==========
joblib.dump(model, "hope_tamil_model.pkl")
print("\n✅ Model saved as hope_tamil_model.pkl")
