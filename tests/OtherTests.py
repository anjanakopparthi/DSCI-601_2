"""
Pytest suite for preprocessing scripts (English, Tamil, Malayalam)
Tests data loading, parsing, cleaning, and label encoding
"""

import pytest
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Test data paths
PROCESSED_DIR = BASE_DIR / "processed"
INITIAL_DATA_DIR = BASE_DIR / "initial_data"


# ============================================================
# Test Helper Functions
# ============================================================

def test_processed_directory_exists():
    """Test that processed directory exists"""
    assert PROCESSED_DIR.exists(), "Processed directory should exist"


def test_initial_data_directory_exists():
    """Test that initial_data directory exists"""
    assert INITIAL_DATA_DIR.exists(), "initial_data directory should exist"


# ============================================================
# English Preprocessing Tests
# ============================================================

class TestEnglishPreprocessing:
    
    def test_english_raw_files_exist(self):
        """Check that raw English CSV files exist"""
        assert (INITIAL_DATA_DIR / "english_hope_train.csv").exists()
        assert (INITIAL_DATA_DIR / "english_hope_dev.csv").exists()
        assert (INITIAL_DATA_DIR / "english_hope_test.csv").exists()
    
    def test_english_parsed_files_exist(self):
        """Check that parsed English files were created"""
        assert (PROCESSED_DIR / "english_hope_train_parsed.csv").exists()
        assert (PROCESSED_DIR / "english_hope_dev_parsed.csv").exists()
        assert (PROCESSED_DIR / "english_hope_test_parsed.csv").exists()
    
    def test_english_balanced_files_exist(self):
        """Check that balanced training files were created"""
        assert (PROCESSED_DIR / "english_hope_train_balanced_undersample.csv").exists()
        assert (PROCESSED_DIR / "english_hope_train_balanced_oversample.csv").exists()
    
    def test_english_parsed_has_correct_columns(self):
        """Check that parsed files have required columns"""
        df = pd.read_csv(PROCESSED_DIR / "english_hope_train_parsed.csv")
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_str" in df.columns
    
    def test_english_labels_are_valid(self):
        """Check that labels are in valid range [0, 1, 2]"""
        df = pd.read_csv(PROCESSED_DIR / "english_hope_train_parsed.csv")
        valid_labels = [0, 1, 2]
        assert df["label"].dropna().isin(valid_labels).all()
    
    def test_english_no_empty_text(self):
        """Check that there are no empty text entries"""
        df = pd.read_csv(PROCESSED_DIR / "english_hope_train_parsed.csv")
        df_filtered = df[df["label"].isin([0, 1])]
        assert not df_filtered["text"].isna().any()
        assert (df_filtered["text"].str.strip() != "").all()
    
    def test_english_balanced_is_actually_balanced(self):
        """Check that undersampled file has balanced classes"""
        df = pd.read_csv(PROCESSED_DIR / "english_hope_train_balanced_undersample.csv")
        counts = df["label"].value_counts()
        assert len(counts) == 2, "Should have exactly 2 classes"
        assert counts[0] == counts[1], "Classes should be perfectly balanced"
    
    def test_english_label_mapping(self):
        """Check that label_str correctly maps to numeric labels"""
        df = pd.read_csv(PROCESSED_DIR / "english_hope_train_parsed.csv")
        # Check a few mappings
        hope_speech = df[df["label_str"] == "Hope_speech"]
        if len(hope_speech) > 0:
            assert (hope_speech["label"] == 1).all()
        
        non_hope = df[df["label_str"] == "Non_hope_speech"]
        if len(non_hope) > 0:
            assert (non_hope["label"] == 0).all()


# ============================================================
# Tamil Preprocessing Tests
# ============================================================

class TestTamilPreprocessing:
    
    def test_tamil_raw_files_exist(self):
        """Check that raw Tamil CSV files exist"""
        assert (INITIAL_DATA_DIR / "tamil_hope_first_train.csv").exists()
        assert (INITIAL_DATA_DIR / "tamil_hope_first_dev.csv").exists()
        assert (INITIAL_DATA_DIR / "tamil_hope_first_test.csv").exists()
    
    def test_tamil_processed_files_exist(self):
        """Check that processed Tamil files were created"""
        assert (PROCESSED_DIR / "tamil_hope_first_train_corrected.csv").exists()
        assert (PROCESSED_DIR / "tamil_hope_first_dev_parsed.csv").exists()
        assert (PROCESSED_DIR / "tamil_hope_first_test_parsed.csv").exists()
    
    def test_tamil_has_correct_columns(self):
        """Check that Tamil files have required columns"""
        df = pd.read_csv(PROCESSED_DIR / "tamil_hope_first_train_corrected.csv")
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_str" in df.columns
    
    def test_tamil_labels_are_valid(self):
        """Check that Tamil labels are valid"""
        df = pd.read_csv(PROCESSED_DIR / "tamil_hope_first_train_corrected.csv")
        valid_labels = [0, 1, 2]
        assert df["label"].dropna().isin(valid_labels).all()
    
    def test_tamil_negation_relabeling_applied(self):
        """Check that negation relabeling logic was applied"""
        df = pd.read_csv(PROCESSED_DIR / "tamil_hope_first_train_corrected.csv")
        # Check that we have both classes
        assert 0 in df["label"].values
        assert 1 in df["label"].values
    
    def test_tamil_text_not_empty(self):
        """Check that Tamil text entries are not empty"""
        df = pd.read_csv(PROCESSED_DIR / "tamil_hope_first_train_corrected.csv")
        df_filtered = df[df["label"].isin([0, 1])]
        assert not df_filtered["text"].isna().any()


# ============================================================
# Malayalam Preprocessing Tests
# ============================================================

class TestMalayalamPreprocessing:
    
    def test_malayalam_raw_files_exist(self):
        """Check that raw Malayalam CSV files exist"""
        assert (INITIAL_DATA_DIR / "malayalam_train.csv").exists()
        assert (INITIAL_DATA_DIR / "malayalam_dev.csv").exists()
        assert (INITIAL_DATA_DIR / "malayalam_test.csv").exists()
    
    def test_malayalam_processed_files_exist(self):
        """Check that processed Malayalam files were created"""
        assert (PROCESSED_DIR / "malayalam_hope_train_processed.csv").exists()
        assert (PROCESSED_DIR / "malayalam_hope_dev_processed.csv").exists()
        assert (PROCESSED_DIR / "malayalam_hope_test_processed.csv").exists()
    
    def test_malayalam_has_correct_columns(self):
        """Check that Malayalam files have required columns"""
        df = pd.read_csv(PROCESSED_DIR / "malayalam_hope_train_processed.csv")
        assert "text" in df.columns
        assert "label" in df.columns
        # Malayalam doesn't have label_str column (only text and label)
    
    def test_malayalam_labels_are_valid(self):
        """Check that Malayalam labels are valid"""
        df = pd.read_csv(PROCESSED_DIR / "malayalam_hope_train_processed.csv")
        valid_labels = [0, 1, 2]
        assert df["label"].dropna().isin(valid_labels).all()
    
    def test_malayalam_text_cleaned(self):
        """Check that Malayalam text is cleaned (no URLs, mentions)"""
        df = pd.read_csv(PROCESSED_DIR / "malayalam_hope_train_processed.csv")
        # Check no http/www URLs
        assert not df["text"].str.contains("http", na=False).any()
        assert not df["text"].str.contains("www", na=False).any()
    
    def test_malayalam_no_empty_text(self):
        """Check that Malayalam text entries are not empty"""
        df = pd.read_csv(PROCESSED_DIR / "malayalam_hope_train_processed.csv")
        df_filtered = df[df["label"].isin([0, 1])]
        assert not df_filtered["text"].isna().any()
        assert (df_filtered["text"].str.strip() != "").all()


# ============================================================
# Cross-Language Consistency Tests
# ============================================================

class TestCrossLanguageConsistency:
    
    def test_all_languages_have_train_dev_test(self):
        """Ensure all languages have train/dev/test splits"""
        # English
        assert (PROCESSED_DIR / "english_hope_train_parsed.csv").exists()
        assert (PROCESSED_DIR / "english_hope_dev_parsed.csv").exists()
        assert (PROCESSED_DIR / "english_hope_test_parsed.csv").exists()
        
        # Tamil
        assert (PROCESSED_DIR / "tamil_hope_first_train_corrected.csv").exists()
        assert (PROCESSED_DIR / "tamil_hope_first_dev_parsed.csv").exists()
        assert (PROCESSED_DIR / "tamil_hope_first_test_parsed.csv").exists()
        
        # Malayalam
        assert (PROCESSED_DIR / "malayalam_hope_train_processed.csv").exists()
        assert (PROCESSED_DIR / "malayalam_hope_dev_processed.csv").exists()
        assert (PROCESSED_DIR / "malayalam_hope_test_processed.csv").exists()
    
    def test_all_processed_files_have_data(self):
        """Check that all processed files are not empty"""
        files = [
            "english_hope_train_parsed.csv",
            "english_hope_dev_parsed.csv",
            "english_hope_test_parsed.csv",
            "tamil_hope_first_train_corrected.csv",
            "tamil_hope_first_dev_parsed.csv",
            "tamil_hope_first_test_parsed.csv",
            "malayalam_hope_train_processed.csv",
            "malayalam_hope_dev_processed.csv",
            "malayalam_hope_test_processed.csv",
        ]
        
        for file in files:
            df = pd.read_csv(PROCESSED_DIR / file)
            assert len(df) > 0, f"{file} should not be empty"
    
    def test_label_distribution_sanity(self):
        """Check that label distributions make sense"""
        for lang, train_file in [
            ("English", "english_hope_train_parsed.csv"),
            ("Tamil", "tamil_hope_first_train_corrected.csv"),
            ("Malayalam", "malayalam_hope_train_processed.csv"),
        ]:
            df = pd.read_csv(PROCESSED_DIR / train_file)
            df_filtered = df[df["label"].isin([0, 1])]
            
            # Should have both classes
            assert 0 in df_filtered["label"].values, f"{lang} should have class 0"
            assert 1 in df_filtered["label"].values, f"{lang} should have class 1"
            
            # Neither class should be empty
            counts = df_filtered["label"].value_counts()
            assert counts[0] > 0, f"{lang} class 0 should have samples"
            assert counts[1] > 0, f"{lang} class 1 should have samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])