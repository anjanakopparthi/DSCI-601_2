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

- models/hope_english_model.pkl
- models/hope_tamil_model.pkl
- models/hope_malayalam_model.pkl


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

### Pytest
- Install libraries in 'Requirements.txt' and run the pytests using the following command
- python ./tests/conftest.py
- python -m pytest tests/otherTests.py -v
- python -m pytest tests/pytestMalayalam.py -vv
- python -m pytest tests/pytestTamil.py -vv
- python -m pytest tests/pytestEnglish.py -vv

### Output 
- The English baseline model achieved an accuracy of 80.4%, with a weighted F1 score of 0.84. However, class-wise performance reveals strong asymmetry. Non-Hope speech (class 0) has very high precision (0.98), indicating that false positives are rare, but Hope speech (class 1) has extremely low precision (0.29), reflecting a high number of misclassified examples. The model overpredicts Hope due to lexical overlap and prior class imbalance in the original training distribution. These observations confirm that the baseline model is insufficient for capturing the nuance and contextual dependencies required for reliable hope speech detection. This motivates the use of transformer-based architectures such as XLM-R to handle code-mixing, negation, and subtle prosocial semantics.

- The Tamil baseline model (TF-IDF + Logistic Regression) achieved an overall accuracy of 63.42%, with a weighted precision of 63.31%, recall of 63.43%, and F1 score of 63.32%. The model performs moderately on both classes, with Non-Hope (0) being detected slightly better (F1 = 0.668) than Hope Speech (1) (F1 = 0.592). The confusion matrix indicates that the model correctly identified 649 out of 946 Non-Hope samples but misclassified 297 as Hope. Similarly, out of 815 Hope samples, 468 were correctly detected, while 347 were misclassified.

- The baseline Malayalam model performed significantly worse than the English and Tamil models, highlighting the difficulty of hope-speech detection in this low-resource setting. The model achieved an accuracy of 30.28%, driven largely by severe class imbalance and limited Malayalam training examples. The precision appears deceptively high (82.34% weighted) because the model overwhelmingly predicts the majority class (label 1, Hope). However, recall is extremely low (30.28% weighted), indicating that the classifier is failing to correctly identify non-hope instances. The confusion matrix confirms this: out of 101 true non-hope samples, the model correctly predicted only 73, while misclassifying 718 non-hope comments as hope. The F1-score (0.38 weighted) further reflects this imbalance and poor generalization. These results suggest that the Malayalam baseline struggles with linguistic sparsity, morphological complexity, and lack of lexical coverage—strong evidence that simple TF-IDF models are insufficient for Malayalam and reinforcing the need for transformer-based contextual models such as XLM-R.