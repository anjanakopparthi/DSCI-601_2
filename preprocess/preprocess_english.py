import pandas as pd
from pathlib import Path
from sklearn.utils import resample

# ============================================================
# 1. Paths to your original raw English CSVs
#    These are the ORIGINAL files with "text;label;" in one column
# ============================================================

base_dir = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed"
base = Path(base_dir)

train_path = base / "english_hope_train.csv"
dev_path   = base / "english_hope_dev.csv"
test_path  = base / "english_hope_test.csv"

# ============================================================
# 2. Helper: split "text;label;" into text + label_str
# ============================================================

def split_text_label(s: str):
    """
    Input example: 'there is hope;Hope_speech;'
    Output: text='there is hope', label_str='Hope_speech'
    """
    parts = str(s).split(';')
    # the last non-empty chunk is the label
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
    """
    Reads a single-column CSV and returns a DataFrame with:
      - text
      - label_str
      - label (numeric)
    """
    raw = pd.read_csv(path, header=None, names=["raw"])
    texts, labels = [], []
    for s in raw["raw"]:
        t, lab = split_text_label(s)
        texts.append(t)
        labels.append(lab)
    df = pd.DataFrame({"text": texts, "label_str": labels})
    return df

# ============================================================
# 3. Parse train / dev / test
# ============================================================

train_df = parse_raw_file(train_path)
dev_df   = parse_raw_file(dev_path)
test_df  = parse_raw_file(test_path)

print("Raw parsed label_str counts (TRAIN):")
print(train_df["label_str"].value_counts(dropna=False))

# ============================================================
# 4. Add numeric labels: 0/1/2
#    0 = Non_hope_speech, 1 = Hope_speech, 2 = not-English
# ============================================================

label_map = {
    "Non_hope_speech": 0,
    "Hope_speech": 1,
    "not-English": 2,
}

for df in (train_df, dev_df, test_df):
    df["label"] = df["label_str"].map(label_map)

print("\nNumeric label counts (TRAIN):")
print(train_df["label"].value_counts(dropna=False))

# ============================================================
# 5. Save PARSED versions (clean)
#    Columns: text, label_str, label
# ============================================================

train_parsed_path = base / "english_hope_train_parsed.csv"
dev_parsed_path   = base / "english_hope_dev_parsed.csv"
test_parsed_path  = base / "english_hope_test_parsed.csv"

train_df.to_csv(train_parsed_path, index=False, encoding="utf-8")
dev_df.to_csv(dev_parsed_path, index=False, encoding="utf-8")
test_df.to_csv(test_parsed_path, index=False, encoding="utf-8")

print("\n Saved parsed files:")
print("  Train parsed:", train_parsed_path)
print("  Dev parsed:  ", dev_parsed_path)
print("  Test parsed: ", test_parsed_path)

# ============================================================
# 6. Create UNDERSAMPLED and OVERSAMPLED train sets
#    Only labels 0/1 (ignore not-English = 2)
# ============================================================

train_bin = train_df[train_df["label"].isin([0, 1])].copy()

print("\nOriginal TRAIN (0/1 only) counts:")
print(train_bin["label"].value_counts())

maj = train_bin[train_bin["label"] == 0]  # Non_hope_speech
minr = train_bin[train_bin["label"] == 1] # Hope_speech

n_maj = len(maj)
n_min = len(minr)
print(f"\nMajority (0=Non_hope_speech): {n_maj}")
print(f"Minority (1=Hope_speech):      {n_min}")

# ---------- A) UNDERSAMPLE majority to minority size ----------
maj_under = resample(
    maj,
    replace=False,
    n_samples=n_min,
    random_state=42
)
train_balanced_under = pd.concat([maj_under, minr], axis=0) \
                         .sample(frac=1.0, random_state=42) \
                         .reset_index(drop=True)

print("\nUndersampled TRAIN counts:")
print(train_balanced_under["label"].value_counts())

# ---------- B) OVERSAMPLE minority to majority size ----------
min_over = resample(
    minr,
    replace=True,
    n_samples=n_maj,
    random_state=42
)
train_balanced_over = pd.concat([maj, min_over], axis=0) \
                        .sample(frac=1.0, random_state=42) \
                        .reset_index(drop=True)

print("\nOversampled TRAIN counts:")
print(train_balanced_over["label"].value_counts())

# ============================================================
# 7. Save balanced train sets
# ============================================================

under_path = base / "english_hope_train_balanced_undersample.csv"
over_path  = base / "english_hope_train_balanced_oversample.csv"

train_balanced_under.to_csv(under_path, index=False, encoding="utf-8")
train_balanced_over.to_csv(over_path, index=False, encoding="utf-8")

print("\n Saved balanced train files:")
print("  Undersampled:", under_path)
print("  Oversampled: ", over_path)
