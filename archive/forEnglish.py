import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load model
model = joblib.load(r"C:/Users/sai pavan preetham a/Desktop/RIT_Anjana/dsci601/DSCI-601_2/models/hope_english_model.pkl")

# Load test dataset
df = pd.read_csv(r"C:/Users/sai pavan preetham a/Desktop/RIT_Anjana/dsci601/DSCI-601_2/processed/english_hope_test_parsed.csv")

# Filter only labels 0 and 1 — baseline model is binary
df = df[df["label"].isin([0,1])]

X_test = df["text"]
y_test = df["label"]

# Predict
preds = model.predict(X_test)

# Metrics for MULTICLASS → use average='weighted'
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds, average="weighted"))
print("Recall:", recall_score(y_test, preds, average="weighted"))
print("F1 Score:", f1_score(y_test, preds, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, preds))
