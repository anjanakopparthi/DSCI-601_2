"""
Multilingual Hope Speech Detection - Data Preprocessing (v2, binary).

Fixes over the previous version:
    1. Rows labeled `not-English` / `not-Tamil` / `not-malayalam` are DELETED
       in every split. Output is strictly binary: 0 = Non_hope_speech,
       1 = Hope_speech, for all three languages.
    2. Strict label encoding — the buggy substring-heuristic fallback is gone.
       Unknown labels are dropped (and counted), never guessed.
    3. Negation-based relabeling of gold training labels is REMOVED.
       (Rule-based logic belongs at prediction time, not in the labels.)
    4. Lighter text cleaning that works for both TF-IDF and XLM-R:
       remove URLs and @mentions, strip the '#' from hashtags (keep the word),
       normalize whitespace. Punctuation, emoji, and casing are PRESERVED
       (XLM-R is a cased model; TfidfVectorizer lowercases on its own).
    5. Explicit deduplication and empty-row removal, with per-split stats.

Usage:
    $ python preprocess_all.py

Reads config.json from the current directory. Expects raw files in
`initial_data/`, writes to `processed_data/` as
`{language}_{split}_processed.csv` with columns: text, label_str, label.
"""

import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

DEFAULT_CONFIG = {
    "initial_data_dir": "initial_data",
    "processed_dir": "processed_data",
    "languages": {
        "english": {
            "train_file": "english_hope_train.csv",
            "dev_file": "english_hope_dev.csv",
            "test_file": "english_hope_test.csv",
            "label_map": {"Hope_speech": 1, "Non_hope_speech": 0},
            "drop_labels": ["not-English"],
        },
        "tamil": {
            "train_file": "tamil_hope_first_train.csv",
            "dev_file": "tamil_hope_first_dev.csv",
            "test_file": "tamil_hope_first_test.csv",
            "label_map": {"Hope_speech": 1, "Non_hope_speech": 0},
            "drop_labels": ["not-Tamil"],
        },
        "malayalam": {
            "train_file": "malayalam_train.csv",
            "dev_file": "malayalam_dev.csv",
            "test_file": "malayalam_test.csv",
            "label_map": {"Hope_speech": 1, "Non_hope_speech": 0},
            "drop_labels": ["not-malayalam"],
        },
    },
}


def load_config(config_path: Path) -> dict:
    """Load config.json; if it does not exist, create it with defaults."""
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"config.json not found — wrote default config to {config_path}")
        return DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Raw file parsing (semicolon single-column format)
# ============================================================

def split_text_label(s: str) -> tuple[str, str | None]:
    """
    Parse one raw row of the form 'text content;Label;'.

    The last non-empty semicolon token is the label; everything before it
    (rejoined with ';') is the text. Returns (text, label_or_None).
    """
    parts = str(s).split(";")
    tokens = [p for p in parts if p != ""]
    if len(tokens) == 0:
        return "", None
    if len(tokens) == 1:
        return tokens[0], None
    return ";".join(tokens[:-1]), tokens[-1].strip()


def parse_raw_file(path: Path) -> pd.DataFrame:
    """Read the raw single-column CSV and split into text / label_str."""
    raw = pd.read_csv(
        path, header=None, names=["raw"], dtype=str,
        skip_blank_lines=True, encoding="utf-8",
    )
    texts, labels = [], []
    for s in raw["raw"].dropna():
        t, lab = split_text_label(s)
        texts.append(t)
        labels.append(lab)
    return pd.DataFrame({"text": texts, "label_str": labels})


# ============================================================
# 3. Text cleaning (light — safe for both TF-IDF and XLM-R)
# ============================================================

URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Light cleaning:
        - remove URLs and @mentions
        - '#hashtag' -> 'hashtag' (keep the word, drop the symbol)
        - collapse whitespace

    Deliberately preserved: punctuation, emoji, casing, and native scripts —
    XLM-R uses all of them; TfidfVectorizer handles lowercasing itself.
    """
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = HASH_RE.sub(r"\1", text)   # unwrap hashtags BEFORE mention removal
    text = MENTION_RE.sub(" ", text)  # so '@#user' doesn't survive as '@user'
    text = WS_RE.sub(" ", text)
    return text.strip()


# ============================================================
# 4. Per-file pipeline
# ============================================================

def preprocess_file(
    input_path: Path,
    lang_name: str,
    lang_cfg: dict,
    output_path: Path,
    split_name: str,
) -> pd.DataFrame:
    """
    Full pipeline for one split:
        parse -> drop not-in-language rows -> clean text ->
        strict label encode -> drop empties/dupes -> save CSV.

    Output columns: text, label_str, label  (label in {0, 1} only).
    """
    print(f"\n=== {lang_name.upper()} — {split_name} ===")
    df = parse_raw_file(input_path)
    n0 = len(df)
    print(f"  Parsed rows:            {n0}")

    # --- drop rows with missing labels ---
    df = df.dropna(subset=["label_str"])
    df["label_str"] = df["label_str"].str.strip()

    # --- DELETE not-in-language rows (case-insensitive, startswith 'not') ---
    drop_labels = {d.lower() for d in lang_cfg.get("drop_labels", [])}
    is_not_lang = (
        df["label_str"].str.lower().isin(drop_labels)
        | df["label_str"].str.lower().str.startswith("not-")
    )
    n_not_lang = int(is_not_lang.sum())
    df = df[~is_not_lang]
    print(f"  Dropped not-in-language: {n_not_lang}")

    # --- strict label encoding: exact match only, drop anything unmapped ---
    label_map = lang_cfg["label_map"]
    df["label"] = df["label_str"].map(label_map)
    n_unmapped = int(df["label"].isna().sum())
    if n_unmapped:
        print(f"  Dropped unmapped labels: {n_unmapped} "
              f"({df.loc[df['label'].isna(), 'label_str'].unique()[:5]})")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # --- clean text ---
    df["text"] = df["text"].apply(clean_text)

    # --- drop empty text ---
    n_before = len(df)
    df = df[df["text"].str.strip() != ""]
    print(f"  Dropped empty text:      {n_before - len(df)}")

    # --- deduplicate exact (text, label) pairs ---
    n_before = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"  Dropped duplicates:      {n_before - len(df)}")

    print(f"  Final rows:              {len(df)}")
    print(f"  Label distribution:      {df['label'].value_counts().to_dict()}"
          f"  (1 = Hope, 0 = Non-hope)")

    df[["text", "label_str", "label"]].to_csv(
        output_path, index=False, encoding="utf-8"
    )
    print(f"  Saved -> {output_path}")
    return df


# ============================================================
# 5. Entry point
# ============================================================

def main():
    # Repo root = one level above the folder this script lives in.
    # Works from any working directory (terminal, VS Code debugger, etc.)
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config(base_dir / "config.json")

    initial_data_dir = base_dir / config["initial_data_dir"]
    processed_dir = base_dir / config["processed_dir"]
    processed_dir.mkdir(exist_ok=True)

    all_stats = []
    for lang_name, lang_cfg in config["languages"].items():
        print("\n" + "=" * 60)
        print(f"Processing language: {lang_name.upper()}")
        print("=" * 60)

        counts = {}
        for split in ("train", "dev", "test"):
            in_path = initial_data_dir / lang_cfg[f"{split}_file"]
            out_path = processed_dir / f"{lang_name}_{split}_processed.csv"
            df = preprocess_file(in_path, lang_name, lang_cfg, out_path, split)
            counts[split] = len(df)
        all_stats.append((lang_name, counts))

    print("\n" + "=" * 60)
    print("SUMMARY: final sample counts (binary: 0 = Non-hope, 1 = Hope)")
    print("=" * 60)
    for lang_name, c in all_stats:
        print(f"{lang_name.capitalize():10s} -> "
              f"train: {c['train']:5d}, dev: {c['dev']:5d}, test: {c['test']:5d}")


if __name__ == "__main__":
    main()