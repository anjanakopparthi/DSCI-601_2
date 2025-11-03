import pandas as pd
import re
import os

# ---------------------------------------------------
# Configuration ( plug in your path and language)
# ---------------------------------------------------
input_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\english_hope_test.csv"
lang = "english"   # change to 'tamil' or 'malayalam' for other datasets


# ---------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------
def clean_text(text, lang):
    text = str(text).strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # Remove emojis and non-text characters (keep language scripts)
    text = re.sub(r"[^\w\s\u0D00-\u0D7F\u0B80-\u0BFF]", "", text)  # Keep Malayalam/Tamil

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # Lowercase only for English
    if lang.lower() == "english":
        text = text.lower()

    return text.strip()


# ---------------------------------------------------
# LABEL ENCODING FUNCTION
# ---------------------------------------------------
def encode_label(label):
    label = str(label).strip().lower()
    if "hope" in label:
        return 1
    else:
        return 0


# ---------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------
def preprocess_file(input_path, lang):
    print(f"\n Processing: {input_path} ({lang})")

    # Read CSV (auto detect delimiter)
    df = pd.read_csv(input_path)
    print("Original shape:", df.shape)

    # Identify possible text and label columns
    text_col = None
    label_col = None
    for col in df.columns:
        if "text" in col.lower():
            text_col = col
        if "label" in col.lower() or "hope" in col.lower():
            label_col = col

    # Fallbacks
    if not text_col:
        text_col = df.columns[0]
    if not label_col:
        label_col = df.columns[-1]

    print(f"Detected columns — text: {text_col}, label: {label_col}")

    # Clean text
    df["text"] = df[text_col].apply(lambda x: clean_text(x, lang))
    df["label"] = df[label_col].apply(encode_label)

    # Drop missing or empty rows
    df = df[df["text"].str.strip().astype(bool)]

    # Save processed file
    out_dir = os.path.join(os.path.dirname(input_path), "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(input_path))

    df[["text", "label"]].to_csv(out_path, index=False)
    print(f" Saved cleaned data to: {out_path}")
    print("Sample:\n", df.head(5).to_string(index=False))

    return df


# ---------------------------------------------------
# Run preprocessing
# ---------------------------------------------------
if __name__ == "__main__":
    preprocess_file(input_path, lang)
