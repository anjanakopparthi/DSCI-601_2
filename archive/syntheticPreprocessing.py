import os
import unicodedata
import pandas as pd
import re

# ==========================
# 1. CONFIG: PATHS
# ==========================
# Change this to your actual processed folder
BASE_DIR = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\processed"

# --- English files ---
EN_TRAIN_REAL = os.path.join(BASE_DIR, "english_hope_train_parsed.csv")
EN_DEV_REAL   = os.path.join(BASE_DIR, "english_hope_dev_parsed.csv")
EN_TEST_REAL  = os.path.join(BASE_DIR, "english_hope_test_parsed.csv")

# Synthetic English (10k) you downloaded from ChatGPT
EN_TRAIN_SYN  = os.path.join(BASE_DIR, "english_hope_synthetic_10000.csv")

# --- Tamil files ---
TA_TRAIN_REAL = os.path.join(BASE_DIR, "tamil_hope_first_train_corrected.csv")
TA_DEV_REAL   = os.path.join(BASE_DIR, "tamil_hope_first_dev_parsed.csv")
TA_TEST_REAL  = os.path.join(BASE_DIR, "tamil_hope_first_test_parsed.csv")

# Synthetic Tamil with native + romanized + code-mixed
TA_TRAIN_SYN  = os.path.join(BASE_DIR, "tamil_hope_synthetic_cm_10000.csv")

# --- Malayalam files ---
ML_TRAIN_REAL = os.path.join(BASE_DIR, "malayalam_train.csv")  # or your parsed version
ML_DEV_REAL   = os.path.join(BASE_DIR, "malayalam_dev.csv")
ML_TEST_REAL  = os.path.join(BASE_DIR, "malayalam_test.csv")

# Synthetic Malayalam with native + romanized + code-mixed
ML_TRAIN_SYN  = os.path.join(BASE_DIR, "malayalam_hope_synthetic_cm_10000.csv")

# Output directory (can be same as BASE_DIR)
OUT_DIR = BASE_DIR

# ==========================
# 2. TEXT NORMALIZATION HELPERS
# ==========================

def normalize_text(s: str) -> str:
    """Basic normalization: NFC, strip, collapse whitespace."""
    if pd.isna(s):
        return ""
    # Unicode normalization (important for Tamil/Malayalam)
    s = unicodedata.normalize("NFC", str(s))
    # Strip and collapse multiple spaces
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning to a dataframe with a 'text' column."""
    df = df.copy()
    df["text"] = df["text"].astype(str).apply(normalize_text)
    # drop empty texts
    df = df[df["text"].str.len() > 0]
    # remove exact duplicates
    df = df.drop_duplicates(subset=["text", "label"], keep="first")
    return df

# ==========================
# 3. SCRIPT DETECTION (TAMIL / MALAYALAM / LATIN)
# ==========================

def has_char_in_range(s: str, start_hex: str, end_hex: str) -> bool:
    start = int(start_hex, 16)
    end = int(end_hex, 16)
    for ch in s:
        if start <= ord(ch) <= end:
            return True
    return False

def has_tamil(s: str) -> bool:
    # Tamil Unicode block: 0B80–0BFF
    return has_char_in_range(s, "0B80", "0BFF")

def has_malayalam(s: str) -> bool:
    # Malayalam Unicode block: 0D00–0D7F
    return has_char_in_range(s, "0D00", "0D7F")

def has_latin(s: str) -> bool:
    # Basic Latin / Latin-1 Supplement as a proxy for romanization
    for ch in s:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return True
    return False

def tag_tamil_variant(text: str) -> str:
    """
    Classify Tamil line into:
    - 'native': Tamil script only
    - 'romanized': Latin only
    - 'mixed': both Tamil script and Latin
    If nothing obvious, fallback 'unknown'.
    """
    t = str(text)
    has_ta = has_tamil(t)
    has_en = has_latin(t)

    if has_ta and not has_en:
        return "native"
    elif not has_ta and has_en:
        return "romanized"
    elif has_ta and has_en:
        return "mixed"
    else:
        return "unknown"

def tag_malayalam_variant(text: str) -> str:
    """
    Same idea for Malayalam.
    """
    t = str(text)
    has_ml = has_malayalam(t)
    has_en = has_latin(t)

    if has_ml and not has_en:
        return "native"
    elif not has_ml and has_en:
        return "romanized"
    elif has_ml and has_en:
        return "mixed"
    else:
        return "unknown"

# ==========================
# 4. LOAD + MERGE + CLEAN PER LANGUAGE
# ==========================

def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'label' in {path}, got {df.columns}")
    return df

# ---------- English ----------

def prepare_english():
    print("=== Preparing English ===")
    train_real = load_csv_safe(EN_TRAIN_REAL)
    train_syn  = load_csv_safe(EN_TRAIN_SYN)
    dev_real   = load_csv_safe(EN_DEV_REAL)
    test_real  = load_csv_safe(EN_TEST_REAL)

    # Keep only hope/non-hope
    train = pd.concat([train_real, train_syn], ignore_index=True)
    train = train[train["label"].isin([0, 1])]
    dev   = dev_real[dev_real["label"].isin([0, 1])].copy()
    test  = test_real[test_real["label"].isin([0, 1])].copy()

    # Clean
    train = clean_dataframe(train)
    dev   = clean_dataframe(dev)
    test  = clean_dataframe(test)

    # Add lang column (optional but nice for XLM-R multi-task)
    train["lang"] = "en"
    dev["lang"]   = "en"
    test["lang"]  = "en"

    # Save
    out_train = os.path.join(OUT_DIR, "english_xlmr_train.csv")
    out_dev   = os.path.join(OUT_DIR, "english_xlmr_dev.csv")
    out_test  = os.path.join(OUT_DIR, "english_xlmr_test.csv")

    train.to_csv(out_train, index=False, encoding="utf-8")
    dev.to_csv(out_dev, index=False, encoding="utf-8")
    test.to_csv(out_test, index=False, encoding="utf-8")

    print("Saved:", out_train)
    print("Saved:", out_dev)
    print("Saved:", out_test)


# ---------- Tamil ----------

def prepare_tamil():
    print("=== Preparing Tamil ===")
    train_real = load_csv_safe(TA_TRAIN_REAL)
    train_syn  = load_csv_safe(TA_TRAIN_SYN)
    dev_real   = load_csv_safe(TA_DEV_REAL)
    test_real  = load_csv_safe(TA_TEST_REAL)

    # Only hope / non-hope
    train = pd.concat([train_real, train_syn], ignore_index=True)
    train = train[train["label"].isin([0, 1])]
    dev   = dev_real[dev_real["label"].isin([0, 1])].copy()
    test  = test_real[test_real["label"].isin([0, 1])].copy()

    # Clean
    train = clean_dataframe(train)
    dev   = clean_dataframe(dev)
    test  = clean_dataframe(test)

    # Variant tagging (native / romanized / mixed)
    train["variant"] = train["text"].apply(tag_tamil_variant)
    dev["variant"]   = dev["text"].apply(tag_tamil_variant)
    test["variant"]  = test["text"].apply(tag_tamil_variant)

    # Language code
    train["lang"] = "ta"
    dev["lang"]   = "ta"
    test["lang"]  = "ta"

    out_train = os.path.join(OUT_DIR, "tamil_xlmr_train.csv")
    out_dev   = os.path.join(OUT_DIR, "tamil_xlmr_dev.csv")
    out_test  = os.path.join(OUT_DIR, "tamil_xlmr_test.csv")

    train.to_csv(out_train, index=False, encoding="utf-8")
    dev.to_csv(out_dev, index=False, encoding="utf-8")
    test.to_csv(out_test, index=False, encoding="utf-8")

    print("Saved:", out_train)
    print("Saved:", out_dev)
    print("Saved:", out_test)


# ---------- Malayalam ----------

def prepare_malayalam():
    print("=== Preparing Malayalam ===")
    train_real = load_csv_safe(ML_TRAIN_REAL)
    train_syn  = load_csv_safe(ML_TRAIN_SYN)
    dev_real   = load_csv_safe(ML_DEV_REAL)
    test_real  = load_csv_safe(ML_TEST_REAL)

    # Only hope / non-hope
    train = pd.concat([train_real, train_syn], ignore_index=True)
    train = train[train["label"].isin([0, 1])]
    dev   = dev_real[dev_real["label"].isin([0, 1])].copy()
    test  = test_real[test_real["label"].isin([0, 1])].copy()

    # Clean
    train = clean_dataframe(train)
    dev   = clean_dataframe(dev)
    test  = clean_dataframe(test)

    # Variant tagging
    train["variant"] = train["text"].apply(tag_malayalam_variant)
    dev["variant"]   = dev["text"].apply(tag_malayalam_variant)
    test["variant"]  = test["text"].apply(tag_malayalam_variant)

    # Language code
    train["lang"] = "ml"
    dev["lang"]   = "ml"
    test["lang"]  = "ml"

    out_train = os.path.join(OUT_DIR, "malayalam_xlmr_train.csv")
    out_dev   = os.path.join(OUT_DIR, "malayalam_xlmr_dev.csv")
    out_test  = os.path.join(OUT_DIR, "malayalam_xlmr_test.csv")

    train.to_csv(out_train, index=False, encoding="utf-8")
    dev.to_csv(out_dev, index=False, encoding="utf-8")
    test.to_csv(out_test, index=False, encoding="utf-8")

    print("Saved:", out_train)
    print("Saved:", out_dev)
    print("Saved:", out_test)


# ==========================
# 5. MAIN
# ==========================

if __name__ == "__main__":
    prepare_english()
    prepare_tamil()
    prepare_malayalam()
    print("✅ All XLM-R ready CSVs created.")
