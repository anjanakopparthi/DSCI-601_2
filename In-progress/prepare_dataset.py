# prepare_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Choose a multilingual transformer model
MODEL_NAME = "xlm-roberta-base"  # works with English, Tamil, Malayalam

# Load cleaned comments
df = pd.read_csv("my_comments_clean.csv")

# Keep only text and label columns
if "label" not in df.columns:
    # If your CSV uses 'hope' / 'not_hope' in another column, rename it
    df = df.rename(columns={"your_label_column": "label"})

df = df[["text_clean", "label"]]

# Split into train and test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Save splits
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Example of tokenizing a batch
sample_texts = train_df["text_clean"].tolist()[:5]
tokens = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
print("Sample tokenized input:", tokens)
