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


# Test data cleaning functions
def test_clean_dataframe_filters_labels():
    df = pd.DataFrame({
        'text': ['hello', 'world', 'test'],
        'label': [0, 1, 2]
    })
    cleaned = clean_dataframe(df, keep_labels=[0, 1])
    assert len(cleaned) == 2
    assert 2 not in cleaned['label'].values


def test_clean_dataframe_removes_empty_text():
    df = pd.DataFrame({
        'text': ['hello', '', None, '   '],
        'label': [0, 1, 0, 1]
    })
    cleaned = clean_dataframe(df, keep_labels=[0, 1])
    assert len(cleaned) == 1


# Test data balancing
def test_balance_train_data_undersample():
    df = pd.DataFrame({
        'text': ['a'] * 100 + ['b'] * 20,
        'label': [0] * 100 + [1] * 20
    })
    balanced = balance_train_data(df, method="undersample", random_state=42)
    assert len(balanced) == 40
    assert balanced['label'].value_counts()[0] == 20
    assert balanced['label'].value_counts()[1] == 20


def test_balance_train_data_oversample():
    df = pd.DataFrame({
        'text': ['a'] * 100 + ['b'] * 20,
        'label': [0] * 100 + [1] * 20
    })
    balanced = balance_train_data(df, method="oversample", random_state=42)
    assert len(balanced) == 200
    assert balanced['label'].value_counts()[0] == 100
    assert balanced['label'].value_counts()[1] == 100


# Test ngram conversion
def test_convert_ngram_range_list_to_tuple():
    params = {"max_features": 5000, "ngram_range": [1, 3]}
    converted = convert_ngram_range(params)
    assert isinstance(converted["ngram_range"], tuple)
    assert converted["ngram_range"] == (1, 3)


def test_convert_ngram_range_already_tuple():
    params = {"max_features": 5000, "ngram_range": (1, 3)}
    converted = convert_ngram_range(params)
    assert converted["ngram_range"] == (1, 3)


# Test model building
def test_build_model_default_params():
    model = build_model()
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert len(model.steps) == 2
    assert model.steps[0][0] == 'tfidf'
    assert model.steps[1][0] == 'clf'


def test_build_model_custom_params():
    tfidf_params = {"max_features": 1000, "ngram_range": [1, 2]}
    model = build_model(tfidf_params=tfidf_params)
    assert model.named_steps['tfidf'].max_features == 1000


# Test rule-based predictor
def test_rule_based_predictor_positive_pattern():
    # Create a simple mock model
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["hope", "optimistic"]
    neg_patterns = ["hopeless", "no hope"]
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["I have hope for tomorrow"]
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1


def test_rule_based_predictor_negative_pattern():
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)
    
    model = MockModel()
    pos_patterns = ["hope", "optimistic"]
    neg_patterns = ["hopeless", "no hope"]
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["there is no hope left"]
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 0


def test_rule_based_predictor_no_pattern_match():
    class MockModel:
        def predict(self, texts):
            return [1] * len(texts)
    
    model = MockModel()
    pos_patterns = ["hope"]
    neg_patterns = ["hopeless"]
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["this is a normal sentence"]
    base, adjusted = predictor(texts)
    
    # Should keep original prediction when no pattern matches
    assert adjusted[0] == base[0]


def test_rule_based_predictor_case_insensitive():
    class MockModel:
        def predict(self, texts):
            return [0] * len(texts)
    
    model = MockModel()
    pos_patterns = ["hope"]
    neg_patterns = []
    
    predictor = create_rule_based_predictor(model, pos_patterns, neg_patterns)
    texts = ["I have HOPE for tomorrow"]
    base, adjusted = predictor(texts)
    
    assert adjusted[0] == 1


# Test train/dev split preparation
def test_prepare_train_dev_split_with_dev():
    train_df = pd.DataFrame({
        'text': ['a', 'b', 'c', 'd'] * 10,
        'label': [0, 1, 0, 1] * 10
    })
    dev_df = pd.DataFrame({
        'text': ['e', 'f'],
        'label': [0, 1]
    })
    
    X_train, X_dev, y_train, y_dev = prepare_train_dev_split(train_df, dev_df)
    
    assert len(X_train) == 40
    assert len(X_dev) == 2
    assert len(y_train) == 40
    assert len(y_dev) == 2


def test_prepare_train_dev_split_without_dev():
    train_df = pd.DataFrame({
        'text': ['a', 'b', 'c', 'd'] * 10,
        'label': [0, 1, 0, 1] * 10
    })
    
    X_train, X_dev, y_train, y_dev = prepare_train_dev_split(
        train_df, 
        dev_df=None, 
        test_size=0.2
    )
    
    assert len(X_train) == 32  # 80% of 40
    assert len(X_dev) == 8     # 20% of 40


def test_prepare_train_dev_split_small_dev():
    train_df = pd.DataFrame({
        'text': ['a', 'b', 'c', 'd'] * 10,
        'label': [0, 1, 0, 1] * 10
    })
    dev_df = pd.DataFrame({
        'text': ['e'],
        'label': [0]
    })
    
    # Should split from train because dev is too small (< 5)
    X_train, X_dev, y_train, y_dev = prepare_train_dev_split(
        train_df, 
        dev_df, 
        min_dev_size=5
    )
    
    assert len(X_dev) > 1  # Should be split from train, not use tiny dev