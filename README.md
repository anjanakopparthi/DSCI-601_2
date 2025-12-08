# Multilingual Hope Speech Detection

**A comprehensive system for detecting hope speech across English, Tamil, and Malayalam languages**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a complete multilingual hope speech detection pipeline supporting English, Tamil, and Malayalam with native script, romanized text, and code-mixed content processing.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Training](#model-training)
- [Testing](#testing)
- [Results](#results)
- [Documentation](#documentation)
- [License](#license)

---

## ✨ Features

- **Multilingual Support**: English, Tamil, and Malayalam
- **Comprehensive Preprocessing**: Text cleaning, normalization, and label encoding
- **Script Handling**: Native scripts, romanized text, and code-mixed content
- **Baseline Models**: TF-IDF + Logistic Regression for each language
- **Rule-Based Enhancement**: Optional pattern-based prediction overrides
- **Automated Testing**: Comprehensive pytest suite for all components
- **Complete Documentation**: Sphinx-generated HTML documentation
- **Inference Ready**: Saved `.pkl` models for deployment

---

## 📁 Project Structure

```
DSCI-601_2/
├── data/                           # Raw datasets
│   ├── english_train.csv
│   ├── english_dev.csv
│   ├── english_test.csv
│   ├── tamil_train.csv
│   ├── tamil_dev.csv
│   ├── tamil_test.csv
│   ├── malayalam_train.csv
│   ├── malayalam_dev.csv
│   └── malayalam_test.csv
│
├── preprocess/                     # Preprocessing module
│   └── preprocess_all_sphinx.py   # Main preprocessing script
│
├── training/                       # Training module
│   └── train_all_sphinx.py        # Main training script
│
├── models/                         # Saved trained models (generated)
│   ├── hope_english_model.pkl
│   ├── hope_tamil_model.pkl
│   └── hope_malayalam_model.pkl
│
├── processed_data/                 # Processed datasets (generated)
│   ├── english_train_processed.csv
│   ├── tamil_train_processed.csv
│   └── malayalam_train_processed.csv
│
├── tests/                          # Test suite
│   ├── pytestEnglish.py
│   ├── pytestTamil.py
│   └── pytestMalayalam.py
│
├── docs/                           # Sphinx documentation
│   ├── source/
│   │   ├── conf.py
│   │   └── index.rst
│   └── build/html/
│       └── index.html
│
├── config.json                     # Configuration file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/multilingual-hope-speech.git
cd multilingual-hope-speech
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
pytest>=7.2.0
```

---

## 📊 Dataset

### Dataset Information

This project uses the **Hope Speech Detection** dataset:

- **English**: 6,000+ samples
- **Tamil**: 5,000+ samples  
- **Malayalam**: 4,500+ samples

### Label Definitions

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Non-Hope | Comments without hopeful content |
| 1 | Hope | Comments expressing hope or encouragement |
| 2 | Not-in-Language | Comments not in target language (filtered out) |

### Data Format

Raw CSV format (semicolon-delimited):
```
text content;Hope_speech;
another text;Non_hope_speech;
```

Processed format:
```csv
text,label_str,label
"i have hope",Hope_speech,1
"this is bad",Non_hope_speech,0
```

### Dataset Access

**Option 1**: If publicly available, download from [source link]

**Option 2**: Contact authors for access

Place downloaded files in the `data/` directory.

---

## 🚀 Quick Start

### Complete Pipeline

```bash
# 1. Preprocess all datasets
python preprocess/preprocess_all_sphinx.py

# 2. Train all models
python training/train_all_sphinx.py

# 3. Run tests
pytest tests/ -v
```

### Expected Runtime

- Preprocessing: ~2-5 minutes
- Training: ~5-10 minutes per language
- Testing: ~1-2 minutes

---

## 🧹 Preprocessing Pipeline

### Configuration

Edit `config.json`:

```json
{
  "initial_data_dir": "data",
  "processed_dir": "processed_data",
  "languages": {
    "english": {
      "train_file": "english_train.csv",
      "dev_file": "english_dev.csv",
      "test_file": "english_test.csv",
      "label_map": {
        "Hope_speech": 1,
        "Non_hope_speech": 0
      },
      "single_column_semicolon": true
    }
  }
}
```

### Preprocessing Steps

1. **Text Cleaning**
   - Remove URLs, mentions, hashtags
   - Normalize whitespace
   - Preserve Unicode scripts (Tamil/Malayalam)

2. **Label Encoding**
   - Convert strings to numeric (0/1)
   - Filter out label 2 (Not-in-Language)

3. **Negation Handling** (Tamil)
   - Detect negation + hope word combinations
   - Relabel incorrectly labeled examples

### Running Preprocessing

```bash
python preprocess/preprocess_all_sphinx.py
```

**Output:**
```
============================================================
Processing language: ENGLISH
============================================================

=== ENGLISH — train ===
Input:  data/english_train.csv
Output: processed_data/english_train_processed.csv
  Original shape: (5000, 2)
  Shape after cleaning: (4523, 3)
  ✓ Saved
```

---

## 🤖 Model Training

### Model Architecture

- **TF-IDF**: 1-3 n-grams, 5000 max features
- **Logistic Regression**: Balanced class weights, L2 regularization

### Training Configuration

In `config.json`:

```json
{
  "models_dir": "models",
  "training": {
    "balance_method": "undersample",
    "tfidf_params": {
      "max_features": 5000,
      "ngram_range": [1, 3]
    },
    "lr_params": {
      "max_iter": 500,
      "class_weight": "balanced"
    }
  }
}
```

### Running Training

```bash
python training/train_all_sphinx.py
```

**Output:**
```
============================================================
Training ENGLISH Model
============================================================

Original distribution:
  Non_hope_speech (0): 3200
  Hope_speech (1):     1323

Balanced distribution (undersample):
1    1323
0    1323

Validation Results:
              precision    recall  f1-score   support
           0      0.850     0.920     0.884       100
           1      0.900     0.810     0.852        90

✓ Model saved to: models/hope_english_model.pkl
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Language-Specific Tests

```bash
# English
pytest tests/pytestEnglish.py -v

# Tamil
pytest tests/pytestTamil.py -v

# Malayalam
pytest tests/pytestMalayalam.py -v
```

### Test Coverage

- Data cleaning and filtering
- Class balancing (under/over sampling)
- Model pipeline building
- Rule-based predictions
- Unicode handling for Tamil/Malayalam

---

## 📈 Results

### English Baseline

| Metric | Value |
|--------|-------|
| Accuracy | 80.4% |
| Weighted F1 | 0.84 |
| Non-Hope Precision | 0.98 |
| Hope Precision | 0.29 |

**Analysis**: High precision for Non-Hope but low for Hope speech. Model overpredicts Hope due to lexical overlap. **Baseline insufficient** - motivates transformer approaches.

### Tamil Baseline

| Metric | Value |
|--------|-------|
| Accuracy | 63.42% |
| Weighted F1 | 63.32% |
| Non-Hope F1 | 0.668 |
| Hope F1 | 0.592 |

**Analysis**: Moderate balanced performance. Non-Hope slightly better detected.

### Malayalam Baseline

| Metric | Value |
|--------|-------|
| Accuracy | 30.28% |
| Weighted Precision | 82.34% |
| Weighted Recall | 30.28% |
| Weighted F1 | 0.38 |

**Analysis**: **Significantly worse** due to:
- Limited training data
- Morphological complexity
- Lack of lexical coverage

**TF-IDF insufficient for Malayalam** - requires transformer models (XLM-R).

### Replicating Results

```bash
# Ensure same data splits and random seeds
python preprocess/preprocess_all_sphinx.py
python training/train_all_sphinx.py

# Results printed during training
```

---

## 📚 Documentation

### Viewing HTML Documentation

```bash
# Windows
start docs/build/html/index.html

# macOS
open docs/build/html/index.html

# Linux
xdg-open docs/build/html/index.html
```

### Documentation Includes

- Complete API reference
- Function parameters and return types
- Usage examples
- Error handling
- Cross-referenced functions

### Rebuilding Documentation

```bash
pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs/source docs/build/html
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

**Anjana Kopparthi** - Lead Developer

---

## 📧 Contact

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/multilingual-hope-speech/issues)

---

## 📚 Citation

```bibtex
@article{kopparthi2024hope,
  title={Multilingual Hope Speech Detection: A Baseline Approach},
  author={Kopparthi, Anjana},
  year={2024}
}
```

---

**⭐ If this project helps your research, please star the repository!**
