# Configuration file for the Sphinx documentation builder.
# Docs for the Multilingual Hope Speech Detection project (DSCI-601).

import sys
from pathlib import Path

# Make the project modules importable for autodoc.
# docs/source/conf.py -> repo root is two levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
for sub in ("preprocess", "training", "evaluation", "data_collection"):
    sys.path.insert(0, str(ROOT / sub))

# -- Project information -----------------------------------------------------
project = "Multilingual Hope Speech Detection"
author = "Anjana Kopparthi"
copyright = "2026, Anjana Kopparthi"
release = "2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# Don't require heavy ML deps to be importable when building docs.
autodoc_mock_imports = ["joblib", "sklearn", "pandas", "numpy", "requests"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
