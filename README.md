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

### 🔧 1. Installation

### Key dependencies

- pandas
- scikit-learn
- joblib
- transformers
- datasets
- torch

### 🧹 2. Preprocessing Pipeline

The raw dataset included three languages:

- English
- Malayalam
- Tamil

### Labels

- 0 — Non-Hope
- 1 — Hope
- 2 — Not-in-Language

### ✔ What the preprocessing scripts do
### a. Basic cleaning

- remove URLs
- remove hashtags & mentions
- collapse whitespace
- normalize Unicode (important for Tamil/Malayalam)
- preserve emojis and expressive signs

### b. Language filtering

Training keeps only labels 0 and 1.
All label 2 (Not-in-Language) rows are discarded.

### Each contains:

- text
- label_str
- label

These are later tokenized for transformer training.

### 🤖 3. Baseline Model Training

Each baseline model uses:

- TF-IDF (1–3 n-grams, 5K features)
- Logistic Regression (balanced)
- Oversampling + undersampling variants

Run baseline English
python baselineModels/trainBaseline_english.py

Tamil
python baselineModels/trainBaseline_tamil.py

Malayalam
python baselineModels/trainBaseline_malayalam.py

### ✔ Output models

- baselineModels/hope_english_model.pkl
- baselineModels/hope_tamil_model.pkl
- baselineModels/hope_malayalam_model.pkl


Each .pkl contains:

- TF-IDF vectorizer
- Logistic Regression classifier

### 🧪 4. Testing the Models

Run the test scripts:

English
python tests/testing_english.py

Tamil
python tests/test_tamil.py

Malayalam
python tests/test_malayalam.py


Each script:

- loads the correct .pkl model
- reads sample inputs
- prints predicted labels

Example output
Input: "there is hope"
Prediction: Hope Speech (1)

Input: "no hope left"
Prediction: Non-Hope (0)
