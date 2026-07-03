"""Shared pytest fixtures for the hope speech project (binary pipeline)."""

import json
from pathlib import Path

import pytest

BASE_PATH = Path(__file__).parent.parent


def load_config():
    config_path = BASE_PATH / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()


@pytest.fixture
def config_data():
    return config


@pytest.fixture
def base_dir():
    return BASE_PATH


@pytest.fixture
def processed_dir():
    return BASE_PATH / config.get("processed_dir", "processed")


@pytest.fixture
def initial_data_dir():
    return BASE_PATH / config.get("initial_data_dir", "initial_data")
