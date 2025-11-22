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

Kept only labels 0 and 1 for training

Removed label 2

c. Variant tagging

For Tamil & Malayalam, added a column:

variant	meaning
native	Tamil/Malayalam script only
romanized	English script representing language
mixed	Code-mixed with English
unknown	fallback
d. Synthetic datasets

10,000 generated samples per language (balanced):

Hope speech

Non-hope speech

Toxic positivity

Code-mixed

Romanized

Native script

Saved in:

processed/*_synthetic_*.csv

e. XLM-R Preprocessed Files

Script preprocessingForXLMR.py produces:

english_xlmr_train.csv

tamil_xlmr_train.csv

malayalam_xlmr_train.csv

Columns included:

text

label_str

label

toxic_positivity

variant

lang

🤖 3. Baseline Model Training

Each baseline model uses:

TF-IDF (1–3 n-grams, 5000 features)

Logistic Regression (balanced class-weights)

Oversampled & undersampled balanced datasets

Run English model:

python baselineModels/trainBaseline_english.py


Tamil:

python baselineModels/trainBaseline_tamil.py


Malayalam:

python baselineModels/trainBaseline_malayalam.py

✔ Output models

baselineModels/hope_english_model.pkl

baselineModels/hope_tamil_model.pkl

baselineModels/hope_malayalam_model.pkl

Each .pkl file contains:

TF-IDF vectorizer

Logistic Regression classifier

Rule-based wrapper (negation override)

🧪 4. Testing the Models

Run:

python tests/test_english.py
python tests/test_tamil.py
python tests/test_malayalam.py


Each script:

Loads the .pkl baseline model

Runs sample sentences

Prints predicted labels

Example output:

Input: "there is hope"
Prediction: Hope Speech (1)

Input: "no hope left"
Prediction: Non-Hope (0)


---

## 🔥 Why this works

✔ Bullet points appear because every list item starts with `-`  
✔ No extra blank lines between bullet points  
✔ No emojis at the start of bullet lines  
✔ Proper fenced code blocks  
✔ Correct indentation for sublists  

---

If you want, I can also **assemble your entire final README.md** with badges, folder tree, install