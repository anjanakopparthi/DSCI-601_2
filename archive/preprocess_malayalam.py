import pandas as pd
import re
from pathlib import Path

# ---------------------------------------------------
# Configuration - Using relative paths
# ---------------------------------------------------
base_dir = Path.cwd()  # Current directory (DSCI-601_2)
initial_data_dir = base_dir / "initial_data"
processed_dir = base_dir / "processed"

# Create processed directory if it doesn't exist
processed_dir.mkdir(exist_ok=True)

# Malayalam dataset paths (without "hope" in filename)
train_path = initial_data_dir / "malayalam_train.csv"
dev_path = initial_data_dir / "malayalam_dev.csv"
test_path = initial_data_dir / "malayalam_test.csv"

lang = "malayalam"

# ---------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------
def clean_text(text, lang):
    text = str(text).strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # Remove emojis and unwanted punctuation (keep Malayalam/Tamil scripts)
    text = re.sub(r"[^\w\s\u0D00-\u0D7F\u0B80-\u0BFF]", "", text)

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
    """
    Encode labels to numeric:
    - Hope_speech -> 1
    - Non_hope_speech -> 0
    - not-Malayalam -> 2
    """
    label = str(label).strip()
    
    label_map = {
        "Hope_speech": 1,
        "Non_hope_speech": 0,
        "not-Malayalam": 2,
    }
    
    # Try exact match first
    if label in label_map:
        return label_map[label]
    
    # Fallback: check if "hope" is in the label (case-insensitive)
    if "hope" in label.lower():
        return 1
    else:
        return 0


# ---------------------------------------------------
# HELPER: Parse semicolon-delimited files
# ---------------------------------------------------
def split_text_label(s: str):
    """
    Input example: 'text content;Hope_speech;'
    Output: text='text content', label_str='Hope_speech'
    """
    parts = str(s).split(';')
    # Remove empty tokens
    tokens = [p for p in parts if p != ""]
    if len(tokens) == 0:
        return "", None
    if len(tokens) == 1:
        # Only text, no label
        return tokens[0], None
    label_str = tokens[-1]
    text = ';'.join(tokens[:-1])
    return text, label_str


def parse_raw_file(path: Path):
    """Parse semicolon-delimited single-column CSV"""
    raw = pd.read_csv(path, header=None, names=["raw"])
    texts, labels = [], []
    for s in raw["raw"]:
        t, lab = split_text_label(s)
        texts.append(t)
        labels.append(lab)
    df = pd.DataFrame({"text": texts, "label_str": labels})
    return df


# ---------------------------------------------------
# MAIN PREPROCESSING FUNCTION
# ---------------------------------------------------
def preprocess_file(input_path, lang, output_path):
    print(f"\n📁 Processing: {input_path.name} ({lang})")

    # Parse the file (handles semicolon-delimited format)
    df = parse_raw_file(input_path)
    print(f"Original shape: {df.shape}")
    print(f"Label distribution (raw):\n{df['label_str'].value_counts(dropna=False)}")

    # Clean text
    df["text"] = df["text"].apply(lambda x: clean_text(x, lang))
    
    # Encode labels
    df["label"] = df["label_str"].apply(encode_label)

    # Drop rows with empty text
    df = df[df["text"].str.strip().astype(bool)]

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])

    print(f"Shape after cleaning: {df.shape}")
    print(f"Label distribution (numeric):\n{df['label'].value_counts()}")

    # Save processed file
    df[["text", "label"]].to_csv(output_path, index=False, encoding="utf-8")
    print(f"✓ Saved to: {output_path}")
    print(f"\nSample (first 3 rows):")
    print(df[["text", "label"]].head(3).to_string(index=False))

    return df


# ---------------------------------------------------
# Run preprocessing on all splits
# ---------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Malayalam Hope Speech Dataset - Preprocessing")
    print("=" * 60)

    # Process train, dev, test
    train_out = processed_dir / "malayalam_hope_train_processed.csv"
    dev_out = processed_dir / "malayalam_hope_dev_processed.csv"
    test_out = processed_dir / "malayalam_hope_test_processed.csv"

    train_df = preprocess_file(train_path, lang, train_out)
    dev_df = preprocess_file(dev_path, lang, dev_out)
    test_df = preprocess_file(test_path, lang, test_out)

    print("\n" + "=" * 60)
    print("✓ Malayalam preprocessing complete!")
    print("=" * 60)
    
    # Summary
    print(f"\nFinal statistics:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")