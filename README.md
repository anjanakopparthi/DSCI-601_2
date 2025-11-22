Multilingual Hope Speech Detection (English • Tamil • Malayalam)

This repository implements a complete multilingual hope-speech detection system for English, Tamil, and Malayalam, supporting native script, romanized text, and code-mixed comments.

### ✔️ Project Features

- Raw dataset ingestion  
- Preprocessing & cleaning  
- Synthetic dataset generation (10K/sample language)  
- Script & variant tagging (native / romanized / mixed)  
- Tokenization for multilingual transformers (XLM-R)  
- Baseline TF-IDF + Logistic Regression models  
- Saved inference-ready `.pkl` models  
- Test scripts to verify predictions  


This work is part of an applied data-science research project on hope speech detection in low-resource and code-mixed environments.

## 🔧 1. Installation

Install all required packages:

```bash
pip install -r requirements.txt



Key dependencies:

pandas

scikit-learn

joblib

transformers

datasets

torch

🧹 2. Preprocessing Pipeline

The raw dataset included three languages:

English

Malayalam

Tamil

Each language contained labels:

0 — Non-Hope

1 — Hope

2 — Not-in-Language

✔ What the preprocessing scripts do:
a. Basic cleaning

Removed URLs

Removed hashtags and mentions

Collapsed whitespace

Normalized Unicode (important for Tamil/Malayalam)

Kept emojis and emotional signals

b. Language filtering

Kept only (0,1) for training, removed label 2.

c. Variant tagging

For Tamil & Malayalam:

Variant	Meaning
native	Tamil/Malayalam script only
romanized	English script but Indian language
mixed	Code-mixed with English
unknown	fallback
d. Synthetic datasets

10,000 generated samples per language:

Hope speech

Non-hope speech

Toxic positivity

Code-mixed

Romanized

Native script

All synthetic data is saved under:

processed/*_synthetic_*.csv

e. Final XLM-R Preprocessed Files

The script preprocessingForXLMR.py produces:

english_xlmr_train.csv
tamil_xlmr_train.csv
malayalam_xlmr_train.csv


These contain:

text

label_str

label

toxic_positivity

variant

lang

These CSVs are then tokenized for transformers training.

🤖 3. Baseline Model Training

Each baseline model is trained on:

TF-IDF (1–3 n-grams, 5K features)

Logistic Regression (balanced class-weights)

Oversampled and undersampled variants

Run English baseline:
python baselineModels/trainBaseline_english.py


Same for:

python baselineModels/trainBaseline_tamil.py
python baselineModels/trainBaseline_malayalam.py

✔ Output

Each script saves a model:

baselineModels/hope_english_model.pkl
baselineModels/hope_tamil_model.pkl
baselineModels/hope_malayalam_model.pkl


These .pkl files contain:

TF-IDF vectorizer

Logistic Regression classifier

Optional rule-based wrapper (negation override, phrase correction)

🧪 4. Testing the Models

To test the predictions using .pkl models:

English:
python tests/test_english.py

Tamil:
python tests/test_tamil.py

Malayalam:
python tests/test_malayalam.py


Each script:

Loads the appropriate .pkl baseline model

Accepts sample user inputs

Prints predicted labels

Example output:

Input: "there is hope"
Prediction: Hope Speech (1)

Input: "no hope left"
Prediction: Non-Hope (0)
