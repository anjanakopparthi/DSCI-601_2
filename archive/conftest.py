import pytest
import joblib
import json
from pathlib import Path

# Base directory (project root)
BASE_PATH = Path(__file__).parent.parent

def load_config():
    """Load config.json from project root"""
    config_path = BASE_PATH / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

# Load configuration
config = load_config()
models_dir = BASE_PATH / config.get("models_dir", "models")

@pytest.fixture
def english_model():
    """Load English base model"""
    model_path = models_dir / "hope_english_model.pkl"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def tamil_model():
    """Load Tamil base model"""
    model_path = models_dir / "hope_tamil_model.pkl"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def malayalam_model():
    """Load Malayalam base model"""
    model_path = models_dir / "hope_malayalam_model.pkl"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def english_model_with_rules():
    """Load English rule-enhanced model"""
    model_path = models_dir / "hope_english_model_with_rules.pkl"
    if not model_path.exists():
        pytest.skip(f"Rule model not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def tamil_model_with_rules():
    """Load Tamil rule-enhanced model"""
    model_path = models_dir / "hope_tamil_model_with_rules.pkl"
    if not model_path.exists():
        pytest.skip(f"Rule model not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def malayalam_model_with_rules():
    """Load Malayalam rule-enhanced model"""
    model_path = models_dir / "hope_malayalam_model_with_rules.pkl"
    if not model_path.exists():
        pytest.skip(f"Rule model not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def config_data():
    """Provide config data to tests"""
    return config

@pytest.fixture
def base_dir():
    """Provide base directory path"""
    return BASE_PATH

@pytest.fixture
def processed_dir():
    """Provide processed directory path"""
    return BASE_PATH / config.get("processed_dir", "processed")

@pytest.fixture
def initial_data_dir():
    """Provide initial data directory path"""
    return BASE_PATH / config.get("initial_data_dir", "initial_data")