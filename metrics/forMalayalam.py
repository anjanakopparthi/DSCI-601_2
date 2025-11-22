import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# -------------------------------------------------
# 1. Load baseline Malayalam model (.pkl)
# -------------------------------------------------
model_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\models\hope_malayalam_model.pkl"
model = joblib.load(model_path)

# -------------------------------------------------
# 2. Load Malayalam TEST dataset (parsed/original)
# -------------------------------------------------
test_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\processed\malayalam_test.csv"
df = pd.read_csv(test_path)

# Keep only hope vs non-hope
df = df[df["label"].isin([0, 1])]

X_test = df["text"]
y_test = df["label"]

# -------------------------------------------------
# 3. Predict
# -------------------------------------------------
preds = model.predict(X_test)

# -------------------------------------------------
# 4. Calculate Metrics
# -------------------------------------------------
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds, average="weighted"))
print("Recall:", recall_score(y_test, preds, average="weighted"))
print("F1 Score:", f1_score(y_test, preds, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, preds))
