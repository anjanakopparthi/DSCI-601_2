import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths to your processed CSVs
train_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\malayalam_train.csv"
dev_path   = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\malayalam_dev.csv"   # if exists
test_path  = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed\malayalam_test.csv"  # if exists

# Load data
train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path) if os.path.exists(dev_path) else None
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

# Train–validation split
X_train = train_df["text"]
y_train = train_df["label"]

if dev_df is not None:
    X_dev = dev_df["text"]
    y_dev = dev_df["label"]
else:
    from sklearn.model_selection import train_test_split
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=200))
])

# Train
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
print("\nValidation Results:")
y_pred = model.predict(X_dev)
print(classification_report(y_dev, y_pred, digits=3))
print("Accuracy:", accuracy_score(y_dev, y_pred))

# Save model
joblib.dump(model, "hope_malayalam_model.pkl")
print("\n Model saved as hope_malayalam_model.pkl")
