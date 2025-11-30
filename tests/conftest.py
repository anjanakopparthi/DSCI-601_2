import pytest
import joblib
import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def english_model():
    return joblib.load(os.path.join(BASE_PATH, "models/hope_english_model.pkl"))

@pytest.fixture
def tamil_model():
    return joblib.load(os.path.join(BASE_PATH, "models/hope_tamil_model.pkl"))

@pytest.fixture
def malayalam_model():
    return joblib.load(os.path.join(BASE_PATH, "models/hope_malayalam_model.pkl"))
