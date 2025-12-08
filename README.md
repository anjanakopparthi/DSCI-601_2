# Multilingual Hope Speech Detection

**A comprehensive system for detecting hope speech across English, Tamil, and Malayalam languages**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a complete multilingual hope speech detection pipeline supporting English, Tamil, and Malayalam with native script, romanized text, and code-mixed content processing.

---

## Table of Contents

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

## Features

- **Multilingual Support**: English, Tamil, and Malayalam
- **Comprehensive Preprocessing**: Text cleaning, normalization, and label encoding
- **Script Handling**: Native scripts, romanized text, and code-mixed content
- **Baseline Models**: TF-IDF + Logistic Regression for each language
- **Rule-Based Enhancement**: Pattern-based prediction overrides for improved accuracy
- **Automated Testing**: Comprehensive pytest suite for all components
- **Complete Documentation**: Sphinx-generated HTML documentation with Google-style docstrings
- **Inference Ready**: Saved `.pkl` models for deployment

---

## Project Structure

```
DSCI-601_2/
├── initial_data/                   # Raw datasets
│   ├── english_hope_train.csv
│   ├── english_hope_dev.csv
│   ├── english_hope_test.csv
│   ├── tamil_hope_first_train.csv
│   ├── tamil_hope_first_dev.csv
│   ├── tamil_hope_first_test.csv
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
│   ├── hope_malayalam_model.pkl
│   ├── hope_english_model_with_rules.pkl
│   ├── hope_tamil_model_with_rules.pkl
│   └── hope_malayalam_model_with_rules.pkl
│
├── processed/                      # Processed datasets (generated)
│   ├── english_train_processed.csv
│   ├── english_dev_processed.csv
│   ├── english_test_processed.csv
│   ├── tamil_train_processed.csv
│   ├── tamil_dev_processed.csv
│   ├── tamil_test_processed.csv
│   ├── malayalam_train_processed.csv
│   ├── malayalam_dev_processed.csv
│   └── malayalam_test_processed.csv
│
├── tests/                          # Test suite
│   ├── pytestEnglish.py           # English model tests
│   ├── pytestTamil.py             # Tamil model tests
│   └── pytestMalayalam.py         # Malayalam model tests
│
├── docs/                           # Sphinx documentation
│   ├── source/
│   │   ├── conf.py                # Sphinx configuration
│   │   └── index.rst              # Documentation source
│   └── build/html/                # Generated HTML docs
│       └── index.html             # Main documentation page
│
├── config.json                     # Configuration file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/DSCI-601_2.git
cd DSCI-601_2
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

See `requirements.txt` for complete list. Key dependencies:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
pytest>=7.2.0
sphinx>=8.0.0
sphinx-rtd-theme>=2.0.0
```

---

## Dataset

### Dataset Information

This project uses the **Hope Speech Detection** dataset containing labeled text samples:

- **English**: 6,000+ samples
- **Tamil**: 5,000+ samples  
- **Malayalam**: 4,500+ samples

### Label Definitions

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Non_hope_speech | Comments without hopeful or encouraging content |
| 1 | Hope_speech | Comments expressing hope, encouragement, or positivity |
| 2 | not-English/not-Tamil/not-Malayalam | Comments not in target language (filtered during training) |

### Data Format

**Raw CSV format** (semicolon-delimited):
```
text content;Hope_speech;
another text;Non_hope_speech;
not in language;not-English;
```

**Processed format** (after preprocessing):
```csv
text,label_str,label
"i have hope for tomorrow",Hope_speech,1
"this situation is terrible",Non_hope_speech,0
```

### Dataset Access

**Source**: [HopeEDI Dataset](https://github.com/bharathichezhiyan/HopeEDI)

The dataset is already included in the `initial_data/` folder of this repository.

---

## Quick Start

### Complete Pipeline (All Languages)

```bash
# 1. Preprocess all datasets
python preprocess/preprocess_all_sphinx.py

# 2. Train all models
python training/train_all_sphinx.py

# 3. Run all tests
pytest tests/ -v
```

### Individual Language Processing

**English:**
```bash
python preprocess/preprocess_all_sphinx.py  # Processes English
python training/train_all_sphinx.py         # Trains English model
pytest tests/pytestEnglish.py -v            # Tests English
```

**Tamil:**
```bash
python preprocess/preprocess_all_sphinx.py  # Processes Tamil
python training/train_all_sphinx.py         # Trains Tamil model
pytest tests/pytestTamil.py -v              # Tests Tamil
```

**Malayalam:**
```bash
python preprocess/preprocess_all_sphinx.py  # Processes Malayalam
python training/train_all_sphinx.py         # Trains Malayalam model
pytest tests/pytestMalayalam.py -v          # Tests Malayalam
```

### Expected Runtime

- Preprocessing: ~2-5 minutes (all languages)
- Training: ~5-10 minutes per language
- Testing: ~1-2 minutes (all tests)

---

## Preprocessing Pipeline

### Configuration

The `config.json` file controls all preprocessing settings:

```json
{
  "initial_data_dir": "initial_data",
  "processed_dir": "processed",
  "models_dir": "models",
  "languages": {
    "english": {
      "train_file": "english_hope_train.csv",
      "dev_file": "english_hope_dev.csv",
      "test_file": "english_hope_test.csv",
      "label_map": {
        "Non_hope_speech": 0,
        "Hope_speech": 1,
        "not-English": 2
      },
      "single_column_semicolon": true,
      "negation": {
        "enabled": false
      }
    }
  }
}
```

### Preprocessing Steps

1. **Text Cleaning**
   - Remove URLs (http://, https://, www.)
   - Remove social media artifacts (@mentions, #hashtags)
   - Normalize whitespace (collapse multiple spaces)
   - Preserve Unicode scripts for Tamil/Malayalam
   - Case normalization (English only)

2. **Label Encoding**
   - Convert string labels to numeric (0/1)
   - Filter out "not-in-language" samples (label 2)
   - Apply label mapping from config

3. **Negation Handling** (Tamil only)
   - Detect negation patterns near hope keywords
   - Pattern: "நம்பிக்கை" (hope) + "இல்லை" (no)
   - Relabel Hope_speech → Non_hope_speech when negation detected
   - Applied only to training data

4. **Output Generation**
   - Save processed CSV files with three columns:
     - `text`: Cleaned text
     - `label_str`: Original label string
     - `label`: Numeric label (0 or 1)

### Running Preprocessing

```bash
python preprocess/preprocess_all_sphinx.py
```

**Expected Output:**
```
============================================================
Processing language: ENGLISH
============================================================

=== ENGLISH — train ===
Input:  initial_data/english_hope_train.csv
Output: processed/english_train_processed.csv
  Original shape: (5000, 2)
  Raw label_str distribution:
Hope_speech        2834
Non_hope_speech    2100
not-English          66

  Negation rule: 0 rows relabeled (not enabled)
  Shape after cleaning: (4523, 3)
  Numeric label distribution:
1    2834
0    1689

  ✓ Saved to processed/english_train_processed.csv
```

---

## Model Training

### Model Architecture

Each baseline model uses:
- **TF-IDF Vectorizer**: 
  - 1-3 n-grams (unigrams, bigrams, trigrams)
  - 5000 max features
  - Captures local text patterns
- **Logistic Regression**: 
  - Balanced class weights
  - L2 regularization
  - Maximum 500 iterations

### Training Configuration

Configured in `config.json`:

```json
{
  "training": {
    "balance_method": "undersample",
    "tfidf_params": {
      "max_features": 5000,
      "ngram_range": [1, 3]
    },
    "lr_params": {
      "max_iter": 500,
      "class_weight": "balanced",
      "n_jobs": -1
    }
  }
}
```

### Training Process

1. **Data Loading**: Load processed CSV files from `processed/`
2. **Class Balancing**: 
   - Undersample majority class to match minority
   - Alternative: oversample minority to match majority
3. **Train/Dev Split**: 
   - Use provided dev set (if available)
   - Otherwise split 20% from training data
4. **Model Training**: 
   - Fit TF-IDF + Logistic Regression pipeline
   - Use balanced class weights
5. **Evaluation**: 
   - Report metrics on dev and test sets
   - Print classification report with precision/recall/F1
6. **Model Saving**: 
   - Base model: `hope_{language}_model.pkl`
   - Rule-enhanced model: `hope_{language}_model_with_rules.pkl`

### Rule-Based Enhancement

The system supports optional rule-based overrides defined in `config.json`:

**English Example:**
```json
"rule_patterns": {
  "positive": [
    "there is hope",
    "i have hope",
    "never lose hope",
    "better days are coming"
  ],
  "negative": [
    "no hope",
    "hopeless",
    "without hope"
  ]
}
```

Rules override model predictions:
- Text matching **positive patterns** → Hope_speech (1)
- Text matching **negative patterns** → Non_hope_speech (0)
- Positive patterns take precedence

### Running Training

```bash
python training/train_all_sphinx.py
```

**Expected Output:**
```
============================================================
HOPE SPEECH BASELINE MODEL TRAINING
============================================================

Configuration:
  Processed data dir: processed
  Models output dir:  models
  Balance method:     undersample
  TF-IDF params:      {'max_features': 5000, 'ngram_range': (1, 3)}
  LogReg params:      {'max_iter': 500, 'class_weight': 'balanced', 'n_jobs': -1}

============================================================
Training ENGLISH Model
============================================================

Loading data:
  Train: processed/english_train_processed.csv
  Dev:   processed/english_dev_processed.csv
  Test:  processed/english_test_processed.csv

Original distribution:
  Non_hope_speech (0): 1689
  Hope_speech (1):     2834

Balanced distribution (undersample):
1    1689
0    1689

Training model...
✓ Model trained!

Validation Results:
              precision    recall  f1-score   support
           0      0.850     0.920     0.884       100
           1      0.900     0.810     0.852        90

    accuracy                          0.868       190

✓ Model saved to: models/hope_english_model.pkl
✓ Rule-based patterns loaded:
  Positive patterns: 12
  Negative patterns: 8
✓ Rule-based model saved to: models/hope_english_model_with_rules.pkl

============================================================
✓ ENGLISH training complete!
============================================================
```

---

## Testing

### Test Suite Overview

Comprehensive pytest suite covering:
- Data cleaning and filtering
- Class balancing (undersampling/oversampling)
- Model pipeline building
- Rule-based predictions
- Unicode handling for Tamil/Malayalam
- Edge cases and error handling

### Running All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with summary
pytest tests/

# Run specific language tests
pytest tests/pytestEnglish.py -v
pytest tests/pytestTamil.py -v
pytest tests/pytestMalayalam.py -v
```

### Test Categories

**1. Data Cleaning Tests**
- Empty text removal
- Label filtering
- Unicode preservation
- Whitespace normalization

**2. Balancing Tests**
- Undersampling validation
- Oversampling validation
- Edge cases (zero samples, imbalanced data)

**3. Model Pipeline Tests**
- Pipeline construction
- TF-IDF parameter conversion
- Model training and prediction

**4. Rule-Based Tests**
- Positive pattern matching
- Negative pattern matching
- Pattern precedence
- Case-insensitive matching

**5. Language-Specific Tests**

**English:**
- Negation detection ("no hope", "hopeless")
- Toxic positivity patterns
- Mixed case handling

**Tamil:**
- Script handling (Tamil Unicode U+0B80-0BFF)
- Negation with inflections
- Hope keyword detection

**Malayalam:**
- Script handling (Malayalam Unicode U+0D00-0D7F)
- Mixed script (Malayalam + emoji)
- Complex morphology

### Example Test Output

```bash
$ pytest tests/pytestEnglish.py -v

====================================== test session starts =======================================
collected 15 items

tests/pytestEnglish.py::test_clean_dataframe_filters_labels PASSED                        [  6%]
tests/pytestEnglish.py::test_clean_dataframe_removes_empty_text PASSED                    [ 13%]
tests/pytestEnglish.py::test_balance_train_data_undersample PASSED                        [ 20%]
tests/pytestEnglish.py::test_balance_train_data_oversample PASSED                         [ 26%]
tests/pytestEnglish.py::test_convert_ngram_range_list_to_tuple PASSED                     [ 33%]
tests/pytestEnglish.py::test_build_model_default_params PASSED                            [ 40%]
tests/pytestEnglish.py::test_rule_based_predictor_positive_pattern PASSED                 [ 46%]
tests/pytestEnglish.py::test_rule_based_predictor_negative_pattern PASSED                 [ 53%]
tests/pytestEnglish.py::test_rule_based_predictor_no_pattern_match PASSED                 [ 60%]
tests/pytestEnglish.py::test_rule_based_predictor_case_insensitive PASSED                 [ 66%]
tests/pytestEnglish.py::test_prepare_train_dev_split_with_dev PASSED                      [ 73%]
tests/pytestEnglish.py::test_prepare_train_dev_split_without_dev PASSED                   [ 80%]
tests/pytestEnglish.py::test_prepare_train_dev_split_small_dev PASSED                     [ 86%]

====================================== 15 passed in 1.56s ================================================
```

---

## Results

### English Baseline Model

| Metric | Value |
|--------|-------|
| **Accuracy** | 80.4% |
| **Weighted F1** | 0.84 |
| **Non-Hope Precision** | 0.98 |
| **Hope Precision** | 0.29 |

**Analysis**: 
- High precision for Non-Hope (98%) indicates few false positives
- Low precision for Hope (29%) indicates many false positives
- Model overpredicts Hope speech due to lexical overlap between classes
- Class imbalance in original data affects performance

**Key Insight**: Baseline TF-IDF model is **insufficient** for capturing nuanced hope speech semantics. This motivates transformer-based approaches (XLM-R, mBERT) which can:
- Capture contextual dependencies
- Handle negation better
- Learn code-mixed patterns
- Understand subtle prosocial semantics

### Tamil Baseline Model

| Metric | Value |
|--------|-------|
| **Accuracy** | 63.42% |
| **Weighted Precision** | 63.31% |
| **Weighted Recall** | 63.43% |
| **Weighted F1** | 63.32% |
| **Non-Hope F1** | 0.668 |
| **Hope F1** | 0.592 |

**Confusion Matrix**:
- Non-Hope: 649/946 correctly identified (68.6%)
- Hope: 468/815 correctly identified (57.4%)
- 297 Non-Hope misclassified as Hope
- 347 Hope misclassified as Non-Hope

**Analysis**: 
- Moderate balanced performance across both classes
- Non-Hope detection slightly better than Hope
- Negation rule improved Tamil performance
- Still struggles with code-mixed Tamil-English text

### Malayalam Baseline Model

| Metric | Value |
|--------|-------|
| **Accuracy** | 30.28% |
| **Weighted Precision** | 82.34% |
| **Weighted Recall** | 30.28% |
| **Weighted F1** | 0.38 |

**Analysis**: 
**Significantly worse** performance highlights major challenges:

1. **Limited Training Data**: Insufficient Malayalam examples for TF-IDF
2. **Morphological Complexity**: Rich morphology not captured by n-grams
3. **Lexical Sparsity**: Limited vocabulary coverage
4. **Class Imbalance**: Model overwhelmingly predicts majority class
5. **Script Complexity**: Malayalam Unicode handling issues

**Confusion Matrix**:
- Only 73/101 Non-Hope correctly identified
- 718 samples misclassified (majority class bias)
- Deceptive high precision due to majority class prediction

**Key Recommendation**: TF-IDF is **insufficient for Malayalam**. Requires:
- Transformer-based contextual models (XLM-R)
- More training data
- Better morphological handling
- Multilingual transfer learning

### Performance Comparison

```
Language      Accuracy    F1-Hope    F1-NonHope    Status
--------      --------    -------    ----------    ------
English       80.4%       0.29       0.98          Baseline inadequate
Tamil         63.4%       0.59       0.67          Moderate performance
Malayalam     30.3%       N/A        N/A           Insufficient - needs transformers
```

### Replicating Results

To replicate the exact results reported:

1. **Use the provided data splits** in `initial_data/`
2. **Keep random seeds consistent** (set in `config.json`)
3. **Run complete pipeline**:

```bash
python preprocess/preprocess_all_sphinx.py
python training/train_all_sphinx.py
```

4. **Results will be printed** during training output
5. **Models saved** in `models/` directory

All results are deterministic given the same data and random seeds.

---

## Documentation

### Viewing HTML Documentation

Complete API documentation with Google-style docstrings:

```bash
# Windows
start docs/build/html/index.html

# macOS
open docs/build/html/index.html

# Linux
xdg-open docs/build/html/index.html
```

### Documentation Contents

- **Module Overview**: System architecture and design
- **Function Reference**: Complete API with all parameters
- **Usage Examples**: Working code snippets
- **Parameter Descriptions**: Detailed type and usage info
- **Return Values**: Expected outputs and types
- **Error Handling**: Exceptions and edge cases
- **Cross-References**: Links between related functions

### Rebuilding Documentation

If you modify code and need to regenerate docs:

```bash
# Install Sphinx (if not already installed)
pip install sphinx sphinx-rtd-theme

# Rebuild HTML documentation
sphinx-build -b html docs/source docs/build/html

# Open updated docs
start docs/build/html/index.html  # Windows
```

### Documentation Standards

This project follows professional documentation practices:
- **Google-style docstrings** for all functions
- **Sphinx** for automated generation
- **Read the Docs theme** for professional appearance
- **Type hints** throughout codebase
- **Comprehensive examples** for each function
- **Cross-referencing** between related components

---

## Research Context

This work is part of an applied data science research project investigating:
- **Hope speech detection** in social media (YouTube comments)
- **Low-resource languages** (Tamil, Malayalam)
- **Code-mixed environments** (native + romanized scripts)
- **Baseline vs. transformer models** performance comparison

### Key Findings

1. **English**: TF-IDF baseline shows promise but limited by lexical overlap
2. **Tamil**: Moderate performance, benefits from negation rules
3. **Malayalam**: Strong evidence that TF-IDF is insufficient for morphologically rich low-resource languages

### Future Work

- Implement transformer-based models (XLM-R, mBERT, IndicBERT)
- Expand dataset with more Malayalam examples
- Handle code-mixed text more robustly
- Develop cross-lingual transfer learning approaches
- Create real-time inference API

---

## License

This project is licensed under the MIT License.

---

## Authors

**Anjana Kopparthi** - Lead Developer

© Copyright 2024, Anjana Kopparthi.  
Built with Sphinx using a theme provided by Read the Docs.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{kopparthi2024hope,
  title={Multilingual Hope Speech Detection: A Baseline Approach},
  author={Kopparthi, Anjana},
  year={2024}
}
```

---

**⭐ If this project helps your research, please star the repository!**