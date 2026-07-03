"""
Tests for the binary preprocessing pipeline (preprocess/preprocess_all.py).

Contract being tested:
    - processed CSVs exist for every language/split with columns
      text, label_str, label
    - labels are STRICTLY binary {0, 1} (not-in-language rows deleted)
    - label_str only contains Hope_speech / Non_hope_speech and maps correctly
    - no empty text, no URL/mention artifacts, no duplicate (text, label) pairs
    - casing/punctuation are PRESERVED (light cleaning for XLM-R)
"""

import pandas as pd
import pytest

LANGS = ["english", "tamil", "malayalam"]
SPLITS = ["train", "dev", "test"]


def _load(processed_dir, lang, split):
    path = processed_dir / f"{lang}_{split}_processed.csv"
    assert path.exists(), f"{path} missing — run preprocess/preprocess_all.py"
    return pd.read_csv(path)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("split", SPLITS)
class TestProcessedData:

    def test_columns(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        assert set(df.columns) == {"text", "label_str", "label"}

    def test_labels_strictly_binary(self, processed_dir, lang, split):
        """Label 2 (not-in-language) must NOT exist anymore."""
        df = _load(processed_dir, lang, split)
        assert set(df["label"].unique()) <= {0, 1}

    def test_no_not_language_label_strings(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        assert not df["label_str"].str.lower().str.startswith("not").any()

    def test_label_string_mapping(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        assert (df.loc[df["label_str"] == "Hope_speech", "label"] == 1).all()
        assert (df.loc[df["label_str"] == "Non_hope_speech", "label"] == 0).all()
        assert set(df["label_str"].unique()) <= {"Hope_speech", "Non_hope_speech"}

    def test_no_empty_text(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        assert not df["text"].isna().any()
        assert (df["text"].astype(str).str.strip() != "").all()

    def test_no_urls_or_mentions(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        text = df["text"].astype(str)
        assert not text.str.contains(r"https?://", regex=True).any()
        assert not text.str.contains(r"www\.", regex=True).any()
        assert not text.str.contains(r"@\w+", regex=True).any()

    def test_no_duplicates(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        assert not df.duplicated(subset=["text", "label"]).any()

    def test_both_classes_present(self, processed_dir, lang, split):
        df = _load(processed_dir, lang, split)
        counts = df["label"].value_counts()
        assert counts.get(0, 0) > 0 and counts.get(1, 0) > 0


class TestCasePreservation:
    """English is no longer lowercased (casing kept for XLM-R)."""

    def test_english_mixed_case_exists(self, processed_dir):
        df = _load(processed_dir, "english", "train")
        has_upper = df["text"].astype(str).str.contains(r"[A-Z]").any()
        assert has_upper, ("expected some uppercase characters — "
                           "English should NOT be lowercased anymore")
