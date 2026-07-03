import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

# ============================
# 1. Load the saved Tamil model
# ============================

model_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\models\hope_tamil_model.pkl"
model = joblib.load(model_path)

# ============================
# 2. Load Tamil test dataset
#    This must be your real parsed test file
# ============================

test_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\processed\tamil_hope_first_test_parsed.csv"
df = pd.read_csv(test_path)

# Keep only 0 and 1 (drop label 2 - not in language)
df = df[df["label"].isin([0, 1])]

X_test = df["text"]
y_test = df["label"]

# ============================
# 3. Predict using model
# ============================

preds = model.predict(X_test)

# ============================
# 4. Metrics
# ============================

print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds, average="weighted"))
print("Recall:", recall_score(y_test, preds, average="weighted"))
print("F1 Score:", f1_score(y_test, preds, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_test, preds, digits=3))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, preds))
