import json
import re
from pathlib import Path
import pandas as pd


# ============================================================
# 1. Load config.json
# ============================================================

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Text cleaning (adapted from your Malayalam script)
#    - keeps Indian scripts, removes URLs, emojis, noise
# ============================================================

def clean_text(text: str, lang: str) -> str:
    text = str(text).strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # Remove emojis and unwanted punctuation
    # Keep word chars, whitespace, Malayalam and Tamil Unicode blocks
    text = re.sub(r"[^\w\s\u0D00-\u0D7F\u0B80-\u0BFF]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # Lowercase only for English
    if lang.lower() == "english":
        text = text.lower()

    return text.strip()


# ============================================================
# 3. Label encoding (generic, uses label_map from config)
#    - close to your previous encode_label logic
# ============================================================

def encode_label(label: str, label_map: dict) -> int:
    label = str(label).strip()

    # Exact match from config
    if label in label_map:
        return label_map[label]

    # Fallback heuristic: use "hope" substring
    if "hope" in label.lower():
        # assume Hope_speech
        for k, v in label_map.items():
            if "hope" in k.lower():
                return v
        return 1
    else:
        # assume Non_hope_speech
        for k, v in label_map.items():
            if "non_hope" in k.lower():
                return v
        return 0


# ============================================================
# 4. Helper: parse "text;label;" single-column format
#    (reused from your Tamil/Malayalam preprocessors)
# ============================================================

def split_text_label(s: str):
    """
    Input example: 'text content;Hope_speech;'
    Returns: text, label_str
    """
    parts = str(s).split(';')
    tokens = [p for p in parts if p != ""]
    if len(tokens) == 0:
        return "", None
    if len(tokens) == 1:
        return tokens[0], None
    label_str = tokens[-1]
    text = ';'.join(tokens[:-1])
    return text, label_str


def parse_raw_file_single_column(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=["raw"])
    texts, labels = [], []
    for s in raw["raw"]:
        t, lab = split_text_label(s)
        texts.append(t)
        labels.append(lab)
    df = pd.DataFrame({"text": texts, "label_str": labels})
    return df


def parse_standard_csv(path: Path,
                       text_col: str = "text",
                       label_col: str = "label_str") -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not in {path}")
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not in {path}")
    return df[[text_col, label_col]].copy()


# ============================================================
# 5. Optional negation relabel rule (for Tamil)
#    - adapted from your tamil correction script
# ============================================================

def apply_negation_rule(df: pd.DataFrame, neg_cfg: dict) -> pd.DataFrame:
    if not neg_cfg.get("enabled", False):
        return df

    hope_word = neg_cfg.get("hope_word", "")
    patterns = neg_cfg.get("patterns", [])
    hope_label_str = neg_cfg.get("hope_label_str", "Hope_speech")
    non_hope_label_str = neg_cfg.get("non_hope_label_str", "Non_hope_speech")

    def has_negation(s: str) -> bool:
        return any(p in str(s) for p in patterns)

    df["has_negation"] = df["text"].astype(str).apply(has_negation)
    df["has_hope_word"] = df["text"].astype(str).str.contains(hope_word, na=False)

    rule_mask = (
        (df["label_str"] == hope_label_str) &
        df["has_negation"] &
        df["has_hope_word"]
    )

    print(f"  Negation rule: {rule_mask.sum()} rows relabeled "
          f"{hope_label_str} → {non_hope_label_str}")

    df.loc[rule_mask, "label_str"] = non_hope_label_str

    # Clean helper cols
    df = df.drop(columns=["has_negation", "has_hope_word"])
    return df


# ============================================================
# 6. Main preprocessing for one file
# ============================================================

def preprocess_file(
    input_path: Path,
    lang_name: str,
    lang_cfg: dict,
    output_path: Path,
    split_name: str
) -> pd.DataFrame:

    print(f"\n=== {lang_name.upper()} – {split_name} ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Choose parser based on config
    if lang_cfg.get("single_column_semicolon", True):
        df = parse_raw_file_single_column(input_path)
    else:
        df = parse_standard_csv(input_path)

    print(f"  Original shape: {df.shape}")
    print(f"  Raw label_str distribution:\n{df['label_str'].value_counts(dropna=False)}")

    # Apply negation rule only to TRAIN if enabled
    if split_name.lower() == "train":
        df = apply_negation_rule(df, lang_cfg.get("negation", {}))

    # Clean text
    df["text"] = df["text"].apply(lambda x: clean_text(x, lang_name))

    # Encode labels
    label_map = lang_cfg.get("label_map", {})
    df["label"] = df["label_str"].apply(lambda l: encode_label(l, label_map))

    # Drop empty text
    df = df[df["text"].str.strip().astype(bool)]

    # Drop NaN labels if any
    df = df.dropna(subset=["label"])

    print(f"  Shape after cleaning: {df.shape}")
    print(f"  Numeric label distribution:\n{df['label'].value_counts(dropna=False)}")

    # FIXED: Save text, label_str, and label columns (not just text + label)
    df[["text", "label_str", "label"]].to_csv(output_path, index=False, encoding="utf-8")

    print(f"  ✓ Saved to {output_path}")
    print(f"  Sample rows:\n{df[['text', 'label_str', 'label']].head(3).to_string(index=False)}")

    return df


# ============================================================
# 7. Orchestrator: loop over all languages + splits
# ============================================================

def main():
    base_dir = Path.cwd()
    config = load_config(base_dir / "config.json")

    initial_data_dir = base_dir / config["initial_data_dir"]
    processed_dir = base_dir / config["processed_dir"]
    processed_dir.mkdir(exist_ok=True)

    all_stats = []

    for lang_name, lang_cfg in config["languages"].items():
        print("\n" + "=" * 60)
        print(f"Processing language: {lang_name.upper()}")
        print("=" * 60)

        # input paths
        train_in = initial_data_dir / lang_cfg["train_file"]
        dev_in   = initial_data_dir / lang_cfg["dev_file"]
        test_in  = initial_data_dir / lang_cfg["test_file"]

        # output paths
        # e.g., english_train_processed.csv, tamil_train_processed.csv
        train_out = processed_dir / f"{lang_name}_train_processed.csv"
        dev_out   = processed_dir / f"{lang_name}_dev_processed.csv"
        test_out  = processed_dir / f"{lang_name}_test_processed.csv"

        # run preprocessing
        train_df = preprocess_file(train_in, lang_name, lang_cfg, train_out, "train")
        dev_df   = preprocess_file(dev_in,   lang_name, lang_cfg, dev_out,   "dev")
        test_df  = preprocess_file(test_in,  lang_name, lang_cfg, test_out,  "test")

        all_stats.append(
            (lang_name, len(train_df), len(dev_df), len(test_df))
        )

    print("\n" + "=" * 60)
    print("SUMMARY: final sample counts")
    print("=" * 60)
    for lang_name, n_train, n_dev, n_test in all_stats:
        print(
            f"{lang_name.capitalize():10s} → "
            f"train: {n_train:5d}, dev: {n_dev:5d}, test: {n_test:5d}"
        )


if __name__ == "__main__":
    main()