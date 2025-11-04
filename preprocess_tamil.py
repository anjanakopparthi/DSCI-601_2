import pandas as pd
from pathlib import Path

# ============================================================
# 1. Paths to your original raw Tamil CSVs
#    (single-column files with "text;label;" format)
# ============================================================

base = Path(r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\project\processed")

train_path = base / "tamil_hope_first_train.csv"
dev_path   = base / "tamil_hope_first_dev.csv"
test_path  = base / "tamil_hope_first_test.csv"

# ============================================================
# 2. Helper: split "text;label;" into text + label_str
# ============================================================

def split_text_label(s: str):
    """
    Input example: 'இது ஒரு நல்ல நாள்;Hope_speech;'
    Output: text='இது ஒரு நல்ல நாள்', label_str='Hope_speech'
    """
    parts = str(s).split(';')
    # Remove empty tokens (because line ends with a ';')
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

print("Raw parsed label_str counts (train):")
print(train_df["label_str"].value_counts(dropna=False))

# ============================================================
# 4. Add numeric labels: 0/1/2
#    0 = Non_hope_speech, 1 = Hope_speech, 2 = not-Tamil
# ============================================================

label_map = {
    "Non_hope_speech": 0,
    "Hope_speech": 1,
    "not-Tamil": 2,
}

for df in (train_df, dev_df, test_df):
    df["label"] = df["label_str"].map(label_map)

# ============================================================
# 5. Apply *relabeling rule* to TRAIN only
#    If text has both "நம்பிக்கை" and a negation word, and is Hope_speech,
#    then flip to Non_hope_speech.
# ============================================================

negation_patterns = ['இல்லை', 'இல்ல', 'கிடையாது', 'மாட்டேன்']
hope_word = 'நம்பிக்கை'

def has_negation(s: str) -> bool:
    return any(p in s for p in negation_patterns)

train_df["has_negation"]   = train_df["text"].astype(str).apply(has_negation)
train_df["has_hope_word"]  = train_df["text"].astype(str).str.contains(hope_word, na=False)

# Rule: Hope_speech + hope_word + negation → relabel to Non_hope_speech
rule_mask = (
    (train_df["label_str"] == "Hope_speech") &
    train_df["has_negation"] &
    train_df["has_hope_word"]
)

print("\nNumber of train rows affected by relabel rule:", rule_mask.sum())

# Apply relabel
train_df.loc[rule_mask, "label_str"] = "Non_hope_speech"
train_df.loc[rule_mask, "label"]     = 0  # Non_hope_speech

# Drop helper columns if you don't want them in final CSVs
train_df = train_df.drop(columns=["has_negation", "has_hope_word"])

# ============================================================
# 6. Save processed files
# ============================================================

train_out = base / "tamil_hope_first_train_corrected.csv"
dev_out   = base / "tamil_hope_first_dev_parsed.csv"
test_out  = base / "tamil_hope_first_test_parsed.csv"

train_df.to_csv(train_out, index=False, encoding="utf-8")
dev_df.to_csv(dev_out, index=False, encoding="utf-8")
test_df.to_csv(test_out, index=False, encoding="utf-8")

print("\n Saved:")
print("  Train (corrected):", train_out)
print("  Dev (parsed):     ", dev_out)
print("  Test (parsed):    ", test_out)

print("\nFinal train label counts (numeric):")
print(train_df["label"].value_counts())
