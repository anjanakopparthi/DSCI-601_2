"""
Multilingual Hope Speech Detection - Data Preprocessing (binary)

What this script does:
1. Reads config.json from the PROJECT ROOT by default.
   Project structure expected:

   DSCI-601_2/
   ├── config.json
   ├── initial_data/
   ├── processed/
   └── preprocess/
       └── preprocess_all.py

2. Uses paths from config.json:
   - base_dir
   - initial_data_dir
   - processed_dir
   - languages -> train/dev/test files

3. Deletes not-in-language rows like:
   - not-English
   - not-Tamil
   - not-Malayalam / not-malayalam

4. Converts labels to binary:
   - Non_hope_speech -> 0
   - Hope_speech     -> 1

5. Saves processed CSVs with columns:
   text, label_str, label

Run from anywhere:
    python preprocess/preprocess_all.py

Or with explicit config:
    python preprocess/preprocess_all.py --config ../config.json
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Config loading and path handling
# ============================================================


def load_config(config_path: Path) -> dict:
    """Load config.json."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at: {config_path}\n"
            "Expected config.json in the project root, e.g. DSCI-601_2/config.json"
        )

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_project_paths(config: dict, config_path: Path) -> tuple[Path, Path, Path]:
    """
    Resolve project paths correctly.

    If config has:
        "base_dir": "."
    then base_dir is resolved relative to the folder containing config.json,
    NOT relative to the preprocess/ script folder.
    """
    config_dir = config_path.parent.resolve()
    base_dir_value = config.get("base_dir", ".")
    base_dir = (config_dir / base_dir_value).resolve()

    initial_data_dir = (base_dir / config["initial_data_dir"]).resolve()
    processed_dir = (base_dir / config["processed_dir"]).resolve()

    return base_dir, initial_data_dir, processed_dir


# ============================================================
# 2. Raw file parsing
# ============================================================


def split_text_label(raw_value: str) -> tuple[str, str | None]:
    """
    Parse one row of this format:
        text content;Label;

    The last non-empty semicolon-separated token is treated as the label.
    Everything before that is treated as text.
    """
    parts = str(raw_value).split(";")
    tokens = [p for p in parts if p != ""]

    if len(tokens) == 0:
        return "", None

    if len(tokens) == 1:
        return tokens[0].strip(), None

    text = ";".join(tokens[:-1]).strip()
    label = tokens[-1].strip()
    return text, label


def parse_raw_file(input_path: Path) -> pd.DataFrame:
    """Read raw CSV and return DataFrame with text and label_str."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(
        input_path,
        header=None,
        names=["raw"],
        dtype=str,
        skip_blank_lines=True,
        encoding="utf-8",
    )

    texts = []
    labels = []

    for raw_value in raw_df["raw"].dropna():
        text, label = split_text_label(raw_value)
        texts.append(text)
        labels.append(label)

    return pd.DataFrame({"text": texts, "label_str": labels})


# ============================================================
# 3. Text cleaning
# ============================================================

URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Light cleaning safe for both TF-IDF and XLM-R.

    Keeps punctuation, casing, emoji, Tamil, Malayalam, etc.
    """
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = HASH_RE.sub(r"\1", text)
    text = MENTION_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


# ============================================================
# 4. Per-split preprocessing
# ============================================================


def preprocess_file(
    input_path: Path,
    output_path: Path,
    lang_name: str,
    split_name: str,
    lang_cfg: dict,
) -> pd.DataFrame:
    """Preprocess one language split and save output CSV."""
    print(f"\n=== {lang_name.upper()} - {split_name.upper()} ===")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    df = parse_raw_file(input_path)
    original_count = len(df)
    print(f"Parsed rows: {original_count}")

    # Drop missing labels
    before = len(df)
    df = df.dropna(subset=["label_str"]).copy()
    df["label_str"] = df["label_str"].astype(str).str.strip()
    print(f"Dropped missing labels: {before - len(df)}")

    # Drop not-in-language rows
    drop_labels = {x.lower().strip() for x in lang_cfg.get("drop_labels", [])}
    label_lower = df["label_str"].str.lower().str.strip()

    is_not_language = label_lower.isin(drop_labels) | label_lower.str.startswith("not-")
    not_language_count = int(is_not_language.sum())
    df = df.loc[~is_not_language].copy()
    print(f"Dropped not-in-language rows: {not_language_count}")

    # Strict label mapping only
    label_map = lang_cfg["label_map"]
    df["label"] = df["label_str"].map(label_map)

    unmapped_count = int(df["label"].isna().sum())
    if unmapped_count:
        bad_labels = df.loc[df["label"].isna(), "label_str"].unique()[:10]
        print(f"Dropped unmapped labels: {unmapped_count}")
        print(f"Unmapped examples: {bad_labels}")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Drop empty text
    before = len(df)
    df = df[df["text"].str.strip() != ""].copy()
    print(f"Dropped empty text rows: {before - len(df)}")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"Dropped duplicate rows: {before - len(df)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label_str", "label"]].to_csv(output_path, index=False, encoding="utf-8")

    print(f"Final rows: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print("Saved successfully")

    return df


# ============================================================
# 5. Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json. If not given, uses ../config.json from this script location.",
    )
    args = parser.parse_args()

    # IMPORTANT FIX:
    # preprocess_all.py is inside DSCI-601_2/preprocess/
    # config.json is inside DSCI-601_2/
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = Path(__file__).resolve().parent.parent / "config.json"

    config = load_config(config_path)
    base_dir, initial_data_dir, processed_dir = get_project_paths(config, config_path)

    print("=" * 70)
    print("PREPROCESSING CONFIG")
    print("=" * 70)
    print(f"Config path      : {config_path}")
    print(f"Base dir         : {base_dir}")
    print(f"Initial data dir : {initial_data_dir}")
    print(f"Processed dir    : {processed_dir}")

    if not initial_data_dir.exists():
        raise FileNotFoundError(
            f"Initial data directory does not exist: {initial_data_dir}\n"
            "Your folder structure should be like:\n"
            "DSCI-601_2/initial_data/\n"
            "DSCI-601_2/processed/\n"
            "DSCI-601_2/preprocess/preprocess_all.py\n"
            "DSCI-601_2/config.json"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    all_counts = []

    for lang_name, lang_cfg in config["languages"].items():
        print("\n" + "=" * 70)
        print(f"PROCESSING LANGUAGE: {lang_name.upper()}")
        print("=" * 70)

        split_counts = {}

        for split_name in ["train", "dev", "test"]:
            file_key = f"{split_name}_file"
            input_file = lang_cfg[file_key]

            input_path = initial_data_dir / input_file
            output_path = processed_dir / f"{lang_name}_{split_name}_processed.csv"

            processed_df = preprocess_file(
                input_path=input_path,
                output_path=output_path,
                lang_name=lang_name,
                split_name=split_name,
                lang_cfg=lang_cfg,
            )

            split_counts[split_name] = len(processed_df)

        all_counts.append((lang_name, split_counts))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Binary labels: 0 = Non_hope_speech, 1 = Hope_speech")

    for lang_name, counts in all_counts:
        print(
            f"{lang_name:10s} -> "
            f"train: {counts['train']:6d}, "
            f"dev: {counts['dev']:6d}, "
            f"test: {counts['test']:6d}"
        )

    print("\nAll processed files saved to:")
    print(processed_dir)


if __name__ == "__main__":
    main()
