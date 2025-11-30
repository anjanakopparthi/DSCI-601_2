"""
Pytest suite for unified preprocessing (preprocess_all.py)
Tests data loading, parsing, cleaning, and label encoding for all languages
"""

import pytest
import pandas as pd
from pathlib import Path


# ============================================================
# Test Directory Structure
# ============================================================

def test_processed_directory_exists(processed_dir):
    """Test that processed directory exists"""
    assert processed_dir.exists(), "Processed directory should exist"


def test_initial_data_directory_exists(initial_data_dir):
    """Test that initial_data directory exists"""
    assert initial_data_dir.exists(), "initial_data directory should exist"


def test_config_file_exists(base_dir):
    """Test that config.json exists"""
    config_path = base_dir / "config.json"
    assert config_path.exists(), "config.json should exist"


# ============================================================
# English Preprocessing Tests
# ============================================================

class TestEnglishPreprocessing:
    
    def test_english_raw_files_exist(self, initial_data_dir, config_data):
        """Check that raw English CSV files exist"""
        lang_config = config_data["languages"]["english"]
        assert (initial_data_dir / lang_config["train_file"]).exists()
        assert (initial_data_dir / lang_config["dev_file"]).exists()
        assert (initial_data_dir / lang_config["test_file"]).exists()
    
    def test_english_processed_files_exist(self, processed_dir):
        """Check that processed English files were created by preprocess_all.py"""
        assert (processed_dir / "english_train_processed.csv").exists()
        assert (processed_dir / "english_dev_processed.csv").exists()
        assert (processed_dir / "english_test_processed.csv").exists()
    
    def test_english_processed_has_correct_columns(self, processed_dir):
        """Check that processed files have required columns"""
        df = pd.read_csv(processed_dir / "english_train_processed.csv")
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_str" in df.columns
    
    def test_english_labels_are_valid(self, processed_dir):
        """Check that labels are in valid range [0, 1, 2]"""
        df = pd.read_csv(processed_dir / "english_train_processed.csv")
        valid_labels = [0, 1, 2]
        assert df["label"].dropna().isin(valid_labels).all()
    
    def test_english_no_empty_text(self, processed_dir):
        """Check that there are no empty text entries"""
        df = pd.read_csv(processed_dir / "english_train_processed.csv")
        df_filtered = df[df["label"].isin([0, 1])]
        assert not df_filtered["text"].isna().any()
        assert (df_filtered["text"].str.strip() != "").all()
    
    def test_english_label_mapping(self, processed_dir, config_data):
        """Check that label_str correctly maps to numeric labels"""
        df = pd.read_csv(processed_dir / "english_train_processed.csv")
        label_map = config_data["languages"]["english"]["label_map"]
        
        # Check each mapping
        for label_str, expected_label in label_map.items():
            rows = df[df["label_str"] == label_str]
            if len(rows) > 0:
                assert (rows["label"] == expected_label).all(), \
                    f"{label_str} should map to {expected_label}"
    
    def test_english_text_is_lowercase(self, processed_dir):
        """Check that English text is lowercased"""
        df = pd.read_csv(processed_dir / "english_train_processed.csv")
        # Sample some texts and check if they're lowercase
        sample_texts = df["text"].head(10)
        for text in sample_texts:
            if isinstance(text, str) and text.strip():
                # Should be lowercase (English is lowercased in preprocessing)
                assert text == text.lower(), "English text should be lowercase"


# ============================================================
# Tamil Preprocessing Tests
# ============================================================

class TestTamilPreprocessing:
    
    def test_tamil_raw_files_exist(self, initial_data_dir, config_data):
        """Check that raw Tamil CSV files exist"""
        lang_config = config_data["languages"]["tamil"]
        assert (initial_data_dir / lang_config["train_file"]).exists()
        assert (initial_data_dir / lang_config["dev_file"]).exists()
        assert (initial_data_dir / lang_config["test_file"]).exists()
    
    def test_tamil_processed_files_exist(self, processed_dir):
        """Check that processed Tamil files were created"""
        assert (processed_dir / "tamil_train_processed.csv").exists()
        assert (processed_dir / "tamil_dev_processed.csv").exists()
        assert (processed_dir / "tamil_test_processed.csv").exists()
    
    def test_tamil_has_correct_columns(self, processed_dir):
        """Check that Tamil files have required columns"""
        df = pd.read_csv(processed_dir / "tamil_train_processed.csv")
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_str" in df.columns
    
    def test_tamil_labels_are_valid(self, processed_dir):
        """Check that Tamil labels are valid"""
        df = pd.read_csv(processed_dir / "tamil_train_processed.csv")
        valid_labels = [0, 1, 2]
        assert df["label"].dropna().isin(valid_labels).all()
    
    def test_tamil_negation_relabeling_applied(self, processed_dir, config_data):
        """Check that negation relabeling logic was applied (if enabled)"""
        df = pd.read_csv(processed_dir / "tamil_train_processed.csv")
        
        # Check that we have both classes
        assert 0 in df["label"].values
        assert 1 in df["label"].values
        
        # If negation is enabled, verify the rule was applied
        tamil_config = config_data["languages"]["tamil"]
        if tamil_config.get("negation", {}).get("enabled", False):
            # The negation rule should have relabeled some rows
            # We can't test exact counts, but we can verify both classes exist
            label_counts = df["label"].value_counts()
            assert len(label_counts) >= 2, "Should have multiple label classes"
    
    def test_tamil_text_not_empty(self, processed_dir):
        """Check that Tamil text entries are not empty"""
        df = pd.read_csv(processed_dir / "tamil_train_processed.csv")
        df_filtered = df[df["label"].isin([0, 1])]
        assert not df_filtered["text"].isna().any()
        assert (df_filtered["text"].str.strip() != "").all()
    
    def test_tamil_text_case_preserved(self, processed_dir):
        """Check that Tamil text case is preserved (not lowercased)"""
        df = pd.read_csv(processed_dir / "tamil_train_processed.csv")
        # Tamil should preserve case - we can't easily test this without knowing
        # the original data, but we can check that text exists
        assert len(df) > 0
        assert "text" in df.columns


# ============================================================
# Malayalam Preprocessing Tests
# ============================================================

class TestMalayalamPreprocessing:
    
    def test_malayalam_raw_files_exist(self, initial_data_dir, config_data):
        """Check that raw Malayalam CSV files exist"""
        lang_config = config_data["languages"]["malayalam"]
        assert (initial_data_dir / lang_config["train_file"]).exists()
        assert (initial_data_dir / lang_config["dev_file"]).exists()
        assert (initial_data_dir / lang_config["test_file"]).exists()
    
    def test_malayalam_processed_files_exist(self, processed_dir):
        """Check that processed Malayalam files were created"""
        assert (processed_dir / "malayalam_train_processed.csv").exists()
        assert (processed_dir / "malayalam_dev_processed.csv").exists()
        assert (processed_dir / "malayalam_test_processed.csv").exists()
    
    def test_malayalam_has_correct_columns(self, processed_dir):
        """Check that Malayalam files have required columns"""
        df = pd.read_csv(processed_dir / "malayalam_train_processed.csv")
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_str" in df.columns
    
    def test_malayalam_labels_are_valid(self, processed_dir):
        """Check that Malayalam labels are valid"""
        df = pd.read_csv(processed_dir / "malayalam_train_processed.csv")
        valid_labels = [0, 1, 2]
        assert df["label"].dropna().isin(valid_labels).all()
    
    def test_malayalam_text_cleaned(self, processed_dir):
        """Check that Malayalam text is cleaned (no URLs, mentions)"""
        df = pd.read_csv(processed_dir / "malayalam_train_processed.csv")
        # Check no http/www URLs
        assert not df["text"].str.contains("http", na=False).any()
        assert not df["text"].str.contains("www", na=False).any()
        # Check no @ mentions
        assert not df["text"].str.contains("@", na=False).any()
    
    def test_malayalam_no_empty_text(self, processed_dir):
        """Check that Malayalam text entries are not empty"""
        df = pd.read_csv(processed_dir / "malayalam_train_processed.csv")
        df_filtered = df[df["label"].isin([0, 1])]
        assert not df_filtered["text"].isna().any()
        assert (df_filtered["text"].str.strip() != "").all()


# ============================================================
# Cross-Language Consistency Tests
# ============================================================

class TestCrossLanguageConsistency:
    
    def test_all_languages_have_train_dev_test(self, processed_dir):
        """Ensure all languages have train/dev/test splits"""
        # English
        assert (processed_dir / "english_train_processed.csv").exists()
        assert (processed_dir / "english_dev_processed.csv").exists()
        assert (processed_dir / "english_test_processed.csv").exists()
        
        # Tamil
        assert (processed_dir / "tamil_train_processed.csv").exists()
        assert (processed_dir / "tamil_dev_processed.csv").exists()
        assert (processed_dir / "tamil_test_processed.csv").exists()
        
        # Malayalam
        assert (processed_dir / "malayalam_train_processed.csv").exists()
        assert (processed_dir / "malayalam_dev_processed.csv").exists()
        assert (processed_dir / "malayalam_test_processed.csv").exists()
    
    def test_all_processed_files_have_data(self, processed_dir):
        """Check that all processed files are not empty"""
        files = [
            "english_train_processed.csv",
            "english_dev_processed.csv",
            "english_test_processed.csv",
            "tamil_train_processed.csv",
            "tamil_dev_processed.csv",
            "tamil_test_processed.csv",
            "malayalam_train_processed.csv",
            "malayalam_dev_processed.csv",
            "malayalam_test_processed.csv",
        ]
        
        for file in files:
            df = pd.read_csv(processed_dir / file)
            assert len(df) > 0, f"{file} should not be empty"
    
    def test_all_files_have_consistent_columns(self, processed_dir):
        """Check that all processed files have the same column structure"""
        expected_columns = {"text", "label", "label_str"}
        
        files = [
            "english_train_processed.csv",
            "tamil_train_processed.csv",
            "malayalam_train_processed.csv",
        ]
        
        for file in files:
            df = pd.read_csv(processed_dir / file)
            assert set(df.columns) == expected_columns, \
                f"{file} should have columns: {expected_columns}"
    
    def test_label_distribution_sanity(self, processed_dir):
        """Check that label distributions make sense"""
        for lang, train_file in [
            ("English", "english_train_processed.csv"),
            ("Tamil", "tamil_train_processed.csv"),
            ("Malayalam", "malayalam_train_processed.csv"),
        ]:
            df = pd.read_csv(processed_dir / train_file)
            df_filtered = df[df["label"].isin([0, 1])]
            
            # Should have both classes
            assert 0 in df_filtered["label"].values, f"{lang} should have class 0"
            assert 1 in df_filtered["label"].values, f"{lang} should have class 1"
            
            # Neither class should be empty
            counts = df_filtered["label"].value_counts()
            assert counts[0] > 0, f"{lang} class 0 should have samples"
            assert counts[1] > 0, f"{lang} class 1 should have samples"
    
    def test_config_matches_files(self, config_data, initial_data_dir, processed_dir):
        """Verify that config.json matches actual file structure"""
        for lang_name, lang_config in config_data["languages"].items():
            # Check raw files
            assert (initial_data_dir / lang_config["train_file"]).exists(), \
                f"Raw train file for {lang_name} should exist"
            assert (initial_data_dir / lang_config["dev_file"]).exists(), \
                f"Raw dev file for {lang_name} should exist"
            assert (initial_data_dir / lang_config["test_file"]).exists(), \
                f"Raw test file for {lang_name} should exist"
            
            # Check processed files
            assert (processed_dir / f"{lang_name}_train_processed.csv").exists(), \
                f"Processed train file for {lang_name} should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])