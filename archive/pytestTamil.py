"""
Tamil Language Tests for Hope Speech Detection Training Module.

Tests Tamil-specific functionality including Unicode handling,
rule-based predictions, negation detection, and data processing pipeline.
"""

import pytest
import sys
import os
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "training")

# Add training folder to import path
sys.path.append(TRAINING_DIR)

from train_all_sphinx import (
    clean_dataframe,
    balance_train_data,
    convert_ngram_range,
    build_model,
    create_rule_based_predictor,
    prepare_train_dev_split
)


# Test Tamil-specific rule patterns
def test_tamil_hope_prediction():
    """Test Tamil hope keyword detection."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)  # Model predicts Non-hope
    
    model = MockModel()
    # Tamil hope patterns
    pos_patterns = ["நம்பிக்கை", "நல்ல", "மகிழ்ச்சி"]  # hope, good, happiness
    neg_patterns = ["இல்லை", "கிடையாது"]  # no, not available
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["நம்பிக்கை இருக்கிறது"]  # "There is hope"
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1, "Expected Hope prediction for Tamil hope keyword"


def test_tamil_negation():
    """Test Tamil negation pattern detection."""
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)  # Model predicts Hope
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை", "நல்ல"]
    neg_patterns = ["இல்லை", "கிடையாது", "இல்ல"]  # no, not available, no (informal)
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    # Text with ONLY negation, no hope keywords
    texts = ["இது சாத்தியமில்லை கிடையாது"]  # "This is not possible at all"
    base, adjusted = predictor(texts)
    
    assert base[0] == 1  # Model predicted Hope
    assert adjusted[0] == 0, "Expected Non-Hope due to Tamil negation"


def test_tamil_complex_negation():
    """Test complex Tamil negation with hope word."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை"]
    neg_patterns = ["இல்லை", "கிடையாது"]
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["என்னிடம் நம்பிக்கை எதுவும் இல்லை"]  # "I have no hope"
    base, adjusted = predictor(texts)
    
    # Positive pattern "நம்பிக்கை" is checked first
    assert adjusted[0] == 1, "Positive pattern checked first in implementation"


def test_tamil_toxic_positivity():
    """Test toxic positivity detection in Tamil."""
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை", "மகிழ்ச்சி"]
    neg_patterns = ["சிரித்து விடு", "எளிது"]  # just smile, easy
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["சிரித்து விடு, அது எளிது"]  # "Just smile, it's easy"
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 0, "Expected Non-Hope for toxic positivity"


def test_tamil_unicode_handling():
    """Test Tamil Unicode with emojis."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["நம்பிக்கை💛"]  # Hope with emoji
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1, "Unicode emojis should not break pattern matching"


def test_tamil_script_handling():
    """Test that Tamil Unicode is preserved in processing."""
    df = pd.DataFrame({
        'text': ['நம்பிக்கை', 'மகிழ்ச்சி', '', 'test'],
        'label': [1, 1, 0, 0]
    })
    
    cleaned = clean_dataframe(df, keep_labels=[0, 1])
    
    # Should keep Tamil text and remove empty strings
    assert len(cleaned) == 3
    assert 'நம்பிக்கை' in cleaned['text'].values
    assert 'மகிழ்ச்சி' in cleaned['text'].values


def test_tamil_mixed_script():
    """Test handling of Tamil + English mix."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை", "hope"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["நம்பிக்கை and hope together"]  # Tamil + English
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1, "Should detect hope in mixed script"


def test_tamil_data_balancing():
    """Test balancing works with Tamil text."""
    df = pd.DataFrame({
        'text': ['நம்பிக்கை'] * 30 + ['இல்லை'] * 120,  # 30 Hope, 120 Non-hope
        'label': [1] * 30 + [0] * 120
    })
    
    balanced = balance_train_data(df, method="undersample", random_state=42)
    
    assert len(balanced) == 60  # Undersampled to minority class size
    assert balanced['label'].value_counts()[0] == 30
    assert balanced['label'].value_counts()[1] == 30


def test_tamil_empty_and_whitespace():
    """Test Tamil text cleaning removes empty/whitespace."""
    df = pd.DataFrame({
        'text': ['நம்பிக்கை', '   ', '', 'மகிழ்ச்சி', None],
        'label': [1, 0, 1, 0, 1]
    })
    
    cleaned = clean_dataframe(df, keep_labels=[0, 1])
    
    assert len(cleaned) == 2
    assert '' not in cleaned['text'].values
    assert None not in cleaned['text'].values


def test_tamil_case_sensitivity():
    """Tamil script doesn't have upper/lowercase, but test pattern matching."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["இது நம்பிக்கையின் செய்தி"]  # "This is a message of hope"
    base, adjusted = predictor(texts)
    
    # Should match even with inflection (நம்பிக்கையின் contains நம்பிக்கை)
    assert adjusted[0] == 1


def test_tamil_model_pipeline():
    """Test that Tamil text can flow through model pipeline."""
    model = build_model(
        tfidf_params={"max_features": 100, "ngram_range": (1, 2)},
        lr_params={"max_iter": 100}
    )
    
    # Create simple Tamil training data
    X_train = pd.Series(['நம்பிக்கை'] * 10 + ['இல்லை'] * 10)
    y_train = pd.Series([1] * 10 + [0] * 10)
    
    # Should not raise errors with Tamil Unicode
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    
    assert len(predictions) == 20
    assert all(p in [0, 1] for p in predictions)


def test_tamil_train_dev_split():
    """Test train/dev split preserves Tamil text."""
    train_df = pd.DataFrame({
        'text': ['நம்பிக்கை', 'மகிழ்ச்சி', 'இல்லை', 'கிடையாது'] * 10,
        'label': [1, 1, 0, 0] * 10
    })
    
    X_train, X_dev, y_train, y_dev = prepare_train_dev_split(
        train_df,
        test_size=0.2,
        random_state=42
    )
    
    # Check Tamil text is preserved
    assert any('நம்பிக்கை' in str(x) for x in X_train.values)
    assert len(X_train) + len(X_dev) == 40


# Additional edge case tests

def test_tamil_only_negation_words():
    """Test text with only negation words (no hope keywords)."""
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை", "மகிழ்ச்சி"]
    neg_patterns = ["இல்லை", "கிடையாது", "இல்ல"]
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["இல்லை இல்லை கிடையாது"]  # Only negation words
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 0, "Should classify as Non-hope"


def test_tamil_empty_patterns():
    """Test predictor with no patterns defined."""
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)
    
    model = MockModel()
    pos_patterns = []
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["நம்பிக்கை"]
    base, adjusted = predictor(texts)
    
    # Should keep model's original prediction
    assert adjusted[0] == base[0]


def test_tamil_special_characters():
    """Test handling of Tamil text with punctuation."""
    df = pd.DataFrame({
        'text': ['நம்பிக்கை!', 'மகிழ்ச்சி?', 'இல்லை.', ''],
        'label': [1, 1, 0, 0]
    })
    
    cleaned = clean_dataframe(df, keep_labels=[0, 1])
    
    # Should preserve Tamil text even with punctuation
    assert len(cleaned) == 3


def test_tamil_oversample_balancing():
    """Test oversampling with Tamil text."""
    df = pd.DataFrame({
        'text': ['நம்பிக்கை'] * 25 + ['இல்லை'] * 100,
        'label': [1] * 25 + [0] * 100
    })
    
    balanced = balance_train_data(df, method="oversample", random_state=42)
    
    assert len(balanced) == 200  # Oversampled to majority class size
    assert balanced['label'].value_counts()[0] == 100
    assert balanced['label'].value_counts()[1] == 100


def test_tamil_multiple_hope_words():
    """Test text with multiple hope-related words."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை", "மகிழ்ச்சி", "நல்ல"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["நம்பிக்கையும் மகிழ்ச்சியும் நல்லது"]  # Multiple hope words
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1, "Should detect hope with multiple keywords"


def test_tamil_negation_inflections():
    """Test various Tamil negation inflections."""
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை"]
    neg_patterns = ["இல்லை", "இல்ல", "இல்லாத", "கிடையாது"]
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    
    # Test different negation forms
    texts = ["இல்லாத சூழ்நிலை"]  # "Non-existent situation"
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 0, "Should detect negation inflection"


def test_tamil_short_text():
    """Test very short Tamil text."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["நம்பிக்கை"]  # Just the hope word
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1


def test_tamil_long_text():
    """Test longer Tamil text with multiple sentences."""
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["நம்பிக்கை", "நல்ல"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["இது ஒரு நல்ல செய்தி. எனக்கு நம்பிக்கை இருக்கிறது. நாம் வெற்றி பெறுவோம்."]
    # "This is good news. I have hope. We will succeed."
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1, "Should detect hope in longer text"