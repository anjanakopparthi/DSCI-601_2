"""
Configuration file for Hope Speech Detection Project
All paths and settings in one place - no hardcoding!
"""

from pathlib import Path

# ============================================================
# PROJECT STRUCTURE
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directories
INITIAL_DATA_DIR = PROJECT_ROOT / "initial_data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
PROCESSED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "english": {
        "raw_files": {
            "train": "english_hope_train.csv",
            "dev": "english_hope_dev.csv",
            "test": "english_hope_test.csv",
        },
        "processed_files": {
            "train": "english_hope_train_parsed.csv",
            "dev": "english_hope_dev_parsed.csv",
            "test": "english_hope_test_parsed.csv",
            "train_balanced_under": "english_hope_train_balanced_undersample.csv",
            "train_balanced_over": "english_hope_train_balanced_oversample.csv",
        },
        "has_label_str": True,
        "language_code": "en",
    },
    "tamil": {
        "raw_files": {
            "train": "tamil_hope_first_train.csv",
            "dev": "tamil_hope_first_dev.csv",
            "test": "tamil_hope_first_test.csv",
        },
        "processed_files": {
            "train": "tamil_hope_first_train_corrected.csv",
            "dev": "tamil_hope_first_dev_parsed.csv",
            "test": "tamil_hope_first_test_parsed.csv",
        },
        "has_label_str": True,
        "language_code": "ta",
        "negation_patterns": ['இல்லை', 'இல்ல', 'கிடையாது', 'மாட்டேன்'],
        "hope_word": 'நம்பிக்கை',
    },
    "malayalam": {
        "raw_files": {
            "train": "malayalam_train.csv",
            "dev": "malayalam_dev.csv",
            "test": "malayalam_test.csv",
        },
        "processed_files": {
            "train": "malayalam_hope_train_processed.csv",
            "dev": "malayalam_hope_dev_processed.csv",
            "test": "malayalam_hope_test_processed.csv",
        },
        "has_label_str": False,
        "language_code": "ml",
    },
}

# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "Non_hope_speech": 0,
    "Hope_speech": 1,
    "not-English": 2,
    "not-Tamil": 2,
    "not-Malayalam": 2,
}

# ============================================================
# PREPROCESSING SETTINGS
# ============================================================

RANDOM_SEED = 42
BALANCE_METHOD = "undersample"  # Options: "undersample", "oversample", "both"

# Text cleaning settings
REMOVE_URLS = True
REMOVE_MENTIONS = True
REMOVE_HASHTAGS = True
REMOVE_EMOJIS = True

# Unicode ranges for different scripts
UNICODE_RANGES = {
    "malayalam": r"\u0D00-\u0D7F",
    "tamil": r"\u0B80-\u0BFF",
}

print(f"✓ Configuration loaded")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  Datasets: {', '.join(DATASETS.keys())}")