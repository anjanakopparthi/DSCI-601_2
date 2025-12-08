"""
Multilingual Hope Speech Detection - Data Preprocessing Module.

This module provides comprehensive preprocessing functionality for hope speech
detection datasets across multiple languages (English, Tamil, Malayalam). It handles
text cleaning, label encoding, negation detection, and generates processed CSV files
ready for model training.

The preprocessing pipeline supports:
    - Multiple input formats (single-column semicolon-delimited and standard CSV)
    - Language-specific text cleaning with Unicode support
    - Configurable label mapping and encoding
    - Negation-based label correction for Tamil
    - Automated processing of train/dev/test splits

Configuration is managed through a config.json file that specifies:
    - Input/output directory paths
    - Language-specific file names
    - Label mappings for each language
    - Negation detection rules

Example:
    Basic usage from command line::

        $ python preprocess_all.py

    This will read config.json and process all configured languages.

Attributes:
    None (module-level attributes are not used)

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""

import json
import re
from pathlib import Path
import pandas as pd


# ============================================================
# 1. Configuration Management
# ============================================================

def load_config(config_path: Path) -> dict:
    """
    Load and parse the JSON configuration file.

    Reads the configuration file that contains all settings for preprocessing,
    including file paths, language-specific parameters, label mappings, and
    negation detection rules.

    Args:
        config_path (Path): Path object pointing to the config.json file.

    Returns:
        dict: Dictionary containing all configuration parameters with structure::

            {
                "initial_data_dir": "initial_data",
                "processed_dir": "processed_data",
                "languages": {
                    "english": {
                        "train_file": "...",
                        "dev_file": "...",
                        "test_file": "...",
                        "label_map": {...},
                        "single_column_semicolon": true,
                        "negation": {...}
                    },
                    ...
                }
            }

    Raises:
        FileNotFoundError: If the configuration file does not exist at the
            specified path.
        json.JSONDecodeError: If the file exists but contains invalid JSON.

    Example:
        >>> config_path = Path("config.json")
        >>> config = load_config(config_path)
        >>> print(config["processed_dir"])
        'processed_data'
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Text Cleaning Functions
# ============================================================

def clean_text(text: str, lang: str) -> str:
    """
    Clean and normalize text for a specific language.

    Performs comprehensive text cleaning including:
        - URL removal (http://, https://, www.)
        - Social media artifact removal (mentions, hashtags)
        - Character filtering (keeps alphanumeric, whitespace, and Indian scripts)
        - Whitespace normalization
        - Case normalization (English only)

    The function preserves Unicode ranges for:
        - Malayalam: U+0D00 to U+0D7F
        - Tamil: U+0B80 to U+0BFF

    Args:
        text (str): Raw input text to be cleaned.
        lang (str): Language identifier. Accepts "english", "tamil", "malayalam".
            Case-insensitive. Only English text is lowercased.

    Returns:
        str: Cleaned and normalized text string. Empty strings and whitespace-only
            strings are stripped to empty string.

    Note:
        - Non-English languages preserve case to maintain script integrity
        - All special characters except language-specific scripts are removed
        - Multiple consecutive spaces are collapsed to single space

    Example:
        >>> text = "Check out https://example.com #hope @user great news!!!"
        >>> clean_text(text, "english")
        'check out great news'

        >>> tamil_text = "நம்பிக்கை @user இருக்கிறது"
        >>> clean_text(tamil_text, "tamil")
        'நம்பிக்கை இருக்கிறது'
    """
    text = str(text).strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # Keep word chars, whitespace, Malayalam and Tamil Unicode blocks
    text = re.sub(r"[^\w\s\u0D00-\u0D7F\u0B80-\u0BFF]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # Lowercase only for English
    if lang.lower() == "english":
        text = text.lower()

    return text.strip()


# ============================================================
# 3. Label Encoding Functions
# ============================================================

def encode_label(label: str, label_map: dict) -> int:
    """
    Encode string labels to numeric values using configured mapping.

    Converts human-readable label strings to integer codes for model training.
    Uses exact matching first, then applies heuristic fallback for variations.

    The function implements a two-tier matching strategy:
        1. Exact match: Direct lookup in provided label_map
        2. Heuristic match: Substring-based detection for common variations

    Args:
        label (str): Raw label string from dataset (e.g., "Hope_speech",
            "Non_hope_speech", "hope", "non-hope").
        label_map (dict): Mapping dictionary from config, e.g.::

            {
                "Hope_speech": 1,
                "Non_hope_speech": 0
            }

    Returns:
        int: Encoded numeric label. Typical values:
            - 0: Non-hope speech
            - 1: Hope speech

    Note:
        Heuristic fallback rules:
            - If "hope" substring found: searches label_map for hope-related key,
              defaults to 1
            - Otherwise: searches for non-hope-related key, defaults to 0

    Warning:
        The heuristic fallback may produce unexpected results if label_map
        keys don't follow standard naming conventions.

    Example:
        >>> label_map = {"Hope_speech": 1, "Non_hope_speech": 0}
        >>> encode_label("Hope_speech", label_map)
        1
        >>> encode_label("hope", label_map)  # Heuristic match
        1
        >>> encode_label("Non_hope_speech", label_map)
        0
    """
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
# 4. File Parsing Functions
# ============================================================

def split_text_label(s: str) -> tuple[str, str | None]:
    """
    Parse semicolon-delimited single-column format.

    Extracts text and label from custom format where each row contains
    text content and label separated by semicolons.

    Format specification::

        text content;Label_value;

    The function handles:
        - Multiple semicolons within text content
        - Trailing semicolons
        - Missing labels (label-only tokens)

    Args:
        s (str): Raw string from single-column CSV format.

    Returns:
        tuple[str, str | None]: A tuple containing:
            - text (str): Extracted text content (may be empty string)
            - label (str | None): Label string, or None if not present

    Note:
        - All tokens before the last non-empty token are considered text
        - The last non-empty token is considered the label
        - Empty strings between semicolons are ignored

    Example:
        >>> split_text_label("This is hope;Hope_speech;")
        ('This is hope', 'Hope_speech')

        >>> split_text_label("Text with;semicolon;Hope_speech;")
        ('Text with;semicolon', 'Hope_speech')

        >>> split_text_label("Only text;")
        ('Only text', None)

        >>> split_text_label(";;")
        ('', None)
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
    """
    Parse CSV file with single-column semicolon-delimited format.

    Reads a CSV file where each row contains a single column with
    semicolon-separated text and label, then splits into structured format.

    Args:
        path (Path): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - text (str): Extracted text content
            - label_str (str): Label string (may contain None values)

    Note:
        Input file should have no headers. The function assigns a temporary
        column name "raw" during processing.

    Example:
        Input CSV content::

            "I have hope for tomorrow;Hope_speech;"
            "This is terrible;Non_hope_speech;"

        Output DataFrame::

                                    text         label_str
            0  I have hope for tomorrow      Hope_speech
            1           This is terrible  Non_hope_speech

    See Also:
        split_text_label: Function used to parse each row
    """
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
    """
    Parse standard CSV file with explicit text and label columns.

    Reads a CSV file with headers and extracts specified columns for
    text and labels. This is the standard format for pre-formatted datasets.

    Args:
        path (Path): Path to the input CSV file.
        text_col (str, optional): Name of the column containing text.
            Defaults to "text".
        label_col (str, optional): Name of the column containing labels.
            Defaults to "label_str".

    Returns:
        pd.DataFrame: DataFrame containing only the text and label columns,
            with original column names preserved.

    Raises:
        KeyError: If either text_col or label_col is not found in the CSV file.

    Example:
        >>> df = parse_standard_csv(Path("data.csv"))
        >>> df.columns.tolist()
        ['text', 'label_str']

        >>> df = parse_standard_csv(
        ...     Path("data.csv"),
        ...     text_col="content",
        ...     label_col="category"
        ... )
        >>> df.columns.tolist()
        ['content', 'category']
    """
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not in {path}")
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not in {path}")
    return df[[text_col, label_col]].copy()


# ============================================================
# 5. Negation Detection and Label Correction
# ============================================================

def apply_negation_rule(df: pd.DataFrame, neg_cfg: dict) -> pd.DataFrame:
    """
    Apply negation-based label correction rules to training data.

    Identifies instances where negation words appear near hope-related keywords
    and relabels them from Hope_speech to Non_hope_speech. This addresses cases
    like "no hope" or "I don't believe in change" that are incorrectly labeled.

    The function is primarily designed for Tamil language data but can be
    configured for other languages through the neg_cfg parameter.

    Args:
        df (pd.DataFrame): Input DataFrame with 'text' and 'label_str' columns.
        neg_cfg (dict): Negation configuration dictionary with structure::

            {
                "enabled": bool,              # Whether to apply rule
                "hope_word": str,             # Hope keyword (e.g., "நம்பிக்கை")
                "patterns": [str, ...],       # Negation patterns to detect
                "hope_label_str": str,        # Label to match (e.g., "Hope_speech")
                "non_hope_label_str": str     # Label to assign (e.g., "Non_hope_speech")
            }

    Returns:
        pd.DataFrame: Modified DataFrame with corrected labels. Original DataFrame
            is not modified; a copy is returned.

    Note:
        - Only rows matching ALL three conditions are relabeled:
            1. Current label matches hope_label_str
            2. Text contains at least one negation pattern
            3. Text contains the hope_word
        - The function prints the number of relabeled rows for verification
        - Temporary helper columns are created and removed during processing

    Warning:
        If neg_cfg["enabled"] is False or not present, the function returns
        the DataFrame unmodified.

    Example:
        >>> neg_cfg = {
        ...     "enabled": True,
        ...     "hope_word": "hope",
        ...     "patterns": ["no", "not", "never"],
        ...     "hope_label_str": "Hope_speech",
        ...     "non_hope_label_str": "Non_hope_speech"
        ... }
        >>> df = apply_negation_rule(df, neg_cfg)
        Negation rule: 15 rows relabeled Hope_speech → Non_hope_speech
    """
    if not neg_cfg.get("enabled", False):
        return df

    hope_word = neg_cfg.get("hope_word", "")
    patterns = neg_cfg.get("patterns", [])
    hope_label_str = neg_cfg.get("hope_label_str", "Hope_speech")
    non_hope_label_str = neg_cfg.get("non_hope_label_str", "Non_hope_speech")

    def has_negation(s: str) -> bool:
        """Check if text contains any negation pattern."""
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
# 6. Main Preprocessing Pipeline
# ============================================================

def preprocess_file(
    input_path: Path,
    lang_name: str,
    lang_cfg: dict,
    output_path: Path,
    split_name: str
) -> pd.DataFrame:
    """
    Execute complete preprocessing pipeline for a single data file.

    This is the main preprocessing function that orchestrates all steps:
        1. Parse input file (format determined by config)
        2. Apply negation rules (training data only)
        3. Clean text (language-specific)
        4. Encode labels to numeric values
        5. Remove invalid entries (empty text, missing labels)
        6. Save processed data to CSV

    Args:
        input_path (Path): Path to raw input CSV file.
        lang_name (str): Language identifier ("english", "tamil", "malayalam").
        lang_cfg (dict): Language-specific configuration from config.json,
            containing file paths, label mappings, and processing rules.
        output_path (Path): Path where processed CSV will be saved.
        split_name (str): Data split identifier ("train", "dev", or "test").
            Negation rules are only applied to "train" split.

    Returns:
        pd.DataFrame: Processed DataFrame with columns:
            - text (str): Cleaned text
            - label_str (str): Original label string
            - label (int): Encoded numeric label

    Note:
        The function prints detailed progress information including:
            - Input/output paths
            - Original and final dataset shapes
            - Label distributions (both string and numeric)
            - Sample rows from processed data

    Example:
        >>> from pathlib import Path
        >>> input_path = Path("raw_data/english_train.csv")
        >>> output_path = Path("processed/english_train_processed.csv")
        >>> lang_cfg = config["languages"]["english"]
        >>> df = preprocess_file(
        ...     input_path, "english", lang_cfg, output_path, "train"
        ... )
        === ENGLISH — train ===
        Input:  raw_data/english_train.csv
        Output: processed/english_train_processed.csv
          Original shape: (5000, 2)
          ...
    """
    print(f"\n=== {lang_name.upper()} — {split_name} ===")
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

    # Save text, label_str, and label columns
    df[["text", "label_str", "label"]].to_csv(output_path, index=False, encoding="utf-8")

    print(f"  ✓ Saved to {output_path}")
    print(f"  Sample rows:\n{df[['text', 'label_str', 'label']].head(3).to_string(index=False)}")

    return df


# ============================================================
# 7. Orchestration and Entry Point
# ============================================================

def main():
    """
    Main execution function - orchestrates preprocessing for all languages.

    This function:
        1. Loads configuration from config.json
        2. Creates output directory structure
        3. Iterates through all configured languages
        4. Processes train/dev/test splits for each language
        5. Prints summary statistics

    The function processes languages in the order they appear in config.json
    and handles all file I/O operations automatically.

    Raises:
        FileNotFoundError: If config.json is not found in current directory.
        KeyError: If required configuration keys are missing.
        Any exceptions from preprocessing steps are propagated.

    Note:
        - Creates processed_dir if it doesn't exist
        - Output files are named: {language}_{split}_processed.csv
        - Prints detailed progress for each language and split
        - Final summary shows sample counts for all languages

    Example:
        Expected console output::

            ============================================================
            Processing language: ENGLISH
            ============================================================

            === ENGLISH — train ===
            Input:  initial_data/english_train.csv
            Output: processed_data/english_train_processed.csv
            ...

            ============================================================
            SUMMARY: final sample counts
            ============================================================
            English    → train:  4523, dev:   500, test:   500
            Tamil      → train:  3891, dev:   450, test:   450
            Malayalam  → train:  4102, dev:   475, test:   475

    See Also:
        preprocess_file: Core preprocessing function called for each file
        load_config: Configuration loading function
    """
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
