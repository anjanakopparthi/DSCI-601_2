"""
Multilingual Hope Speech Detection - Model Training Module.

This module provides comprehensive training functionality for hope speech
detection models across multiple languages (English, Tamil, Malayalam). It handles
data balancing, model training, evaluation, and rule-based prediction enhancements.

The training pipeline supports:
    - Multiple languages with unified pipeline
    - Data balancing (undersampling/oversampling)
    - TF-IDF feature extraction with configurable parameters
    - Logistic Regression classification
    - Rule-based prediction overrides
    - Automated train/dev/test evaluation
    - Model persistence with joblib

Configuration is managed through a config.json file that specifies:
    - Processed data directory paths
    - Model output directory
    - Training parameters (balance method, TF-IDF settings, etc.)
    - Language-specific rule patterns

Example:
    Basic usage from command line::

        $ python train_all.py

    This will read config.json and train models for all configured languages.

Attributes:
    None (module-level attributes are not used)

.. _Scikit-learn Documentation:
   https://scikit-learn.org/stable/documentation.html
"""

import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ============================================================
# 1. Configuration Management
# ============================================================

def load_config(config_path: Path) -> dict:
    """
    Load and parse the JSON configuration file.

    Reads the configuration file that contains all settings for model training,
    including file paths, training parameters, and language-specific settings.

    Args:
        config_path (Path): Path object pointing to the config.json file.

    Returns:
        dict: Dictionary containing all configuration parameters with structure::

            {
                "processed_dir": "processed_data",
                "models_dir": "models",
                "training": {
                    "balance_method": "undersample",
                    "tfidf_params": {...},
                    "lr_params": {...}
                },
                "languages": {...}
            }

    Raises:
        FileNotFoundError: If the configuration file does not exist at the
            specified path.
        json.JSONDecodeError: If the file exists but contains invalid JSON.

    Example:
        >>> config_path = Path("config.json")
        >>> config = load_config(config_path)
        >>> print(config["models_dir"])
        'models'
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Data Cleaning and Filtering
# ============================================================

def clean_dataframe(df: pd.DataFrame, keep_labels: list = [0, 1]) -> pd.DataFrame:
    """
    Clean and filter DataFrame for binary classification.

    Removes invalid entries and ensures data quality by:
        - Filtering to specified label values only
        - Removing rows with empty text
        - Converting text to string type
        - Handling missing values

    Args:
        df (pd.DataFrame): Input DataFrame with 'text' and 'label' columns.
        keep_labels (list, optional): List of label values to retain.
            Defaults to [0, 1] for binary classification.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only valid entries.

    Note:
        - Original DataFrame is not modified (uses copy)
        - Empty strings after stripping whitespace are removed
        - Missing text values are filled with empty string before filtering

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['hello', '', 'world', None],
        ...     'label': [1, 0, 1, 2]
        ... })
        >>> clean_df = clean_dataframe(df, keep_labels=[0, 1])
        >>> len(clean_df)
        2
    """
    df = df.copy()
    df = df[df["label"].isin(keep_labels)]
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    return df


# ============================================================
# 3. Data Balancing
# ============================================================

def balance_train_data(df: pd.DataFrame, method: str = "undersample", random_state: int = 42) -> pd.DataFrame:
    """
    Balance training data to handle class imbalance.

    Addresses class imbalance using either undersampling (reducing majority class)
    or oversampling (increasing minority class) techniques. Prints before/after
    statistics for verification.

    The function assumes binary classification with:
        - Label 0: Non_hope_speech (typically majority)
        - Label 1: Hope_speech (typically minority)

    Args:
        df (pd.DataFrame): Input DataFrame with 'label' column containing 0s and 1s.
        method (str, optional): Balancing strategy. Options:
            - "undersample": Reduce majority class to match minority
            - "oversample": Increase minority class to match majority
            Defaults to "undersample".
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.

    Returns:
        pd.DataFrame: Balanced DataFrame with equal class distribution,
            shuffled randomly.

    Raises:
        ValueError: If method is not "undersample" or "oversample".

    Warning:
        If either class has 0 samples, balancing cannot be performed and
        the original DataFrame is returned with a warning message.

    Note:
        - Undersampling: May lose information but prevents overfitting
        - Oversampling: Uses all data but may cause overfitting on minority class
        - Result is always shuffled to prevent ordering bias

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['a']*100 + ['b']*20,
        ...     'label': [0]*100 + [1]*20
        ... })
        >>> balanced = balance_train_data(df, method="undersample")
        Original distribution:
          Non_hope_speech (0): 100
          Hope_speech (1):     20
        
        Balanced distribution (undersample):
        1    20
        0    20
        >>> len(balanced)
        40
    """
    maj = df[df["label"] == 0]  # Non_hope_speech
    minr = df[df["label"] == 1]  # Hope_speech
    
    n_maj = len(maj)
    n_min = len(minr)
    
    print(f"\nOriginal distribution:")
    print(f"  Non_hope_speech (0): {n_maj}")
    print(f"  Hope_speech (1):     {n_min}")
    
    if n_maj == 0 or n_min == 0:
        print("WARNING: Cannot balance - one class has 0 samples!")
        return df
    
    if method == "undersample":
        # Undersample majority to match minority
        maj_sampled = resample(maj, replace=False, n_samples=n_min, random_state=random_state)
        balanced_df = pd.concat([maj_sampled, minr])
    elif method == "oversample":
        # Oversample minority to match majority
        min_sampled = resample(minr, replace=True, n_samples=n_maj, random_state=random_state)
        balanced_df = pd.concat([maj, min_sampled])
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    print(f"\nBalanced distribution ({method}):")
    print(balanced_df["label"].value_counts())
    
    return balanced_df


# ============================================================
# 4. Train/Dev Split Preparation
# ============================================================

def prepare_train_dev_split(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame = None,
    test_size: float = 0.2,
    random_state: int = 42,
    min_dev_size: int = 5
):
    """
    Prepare training and development sets for model training.

    Creates train/dev split using either a provided dev set or by splitting
    the training data. Handles edge cases like small dev sets and stratification
    failures gracefully.

    Args:
        train_df (pd.DataFrame): Training DataFrame with 'text' and 'label' columns.
        dev_df (pd.DataFrame, optional): Development/validation DataFrame.
            If None or too small, will split from train_df. Defaults to None.
        test_size (float, optional): Proportion of train_df to use for dev
            if dev_df is not provided. Defaults to 0.2 (20%).
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
        min_dev_size (int, optional): Minimum acceptable dev set size.
            If dev_df is smaller, will split from train instead. Defaults to 5.

    Returns:
        tuple: Four pandas Series objects:
            - X_train (pd.Series): Training texts
            - X_dev (pd.Series): Development texts
            - y_train (pd.Series): Training labels
            - y_dev (pd.Series): Development labels

    Note:
        - Attempts stratified split first to preserve class distribution
        - Falls back to non-stratified split if stratification fails
          (e.g., when a class has too few samples)
        - Prints diagnostic information about split strategy and sizes

    Example:
        >>> train_df = pd.DataFrame({
        ...     'text': ['a', 'b', 'c', 'd'],
        ...     'label': [0, 1, 0, 1]
        ... })
        >>> X_train, X_dev, y_train, y_dev = prepare_train_dev_split(train_df)
        Dev set too small or missing → splitting from train
          Using stratified split
        
        Final dataset sizes:
          Train: 3
          Dev:   1
    """
    X_train = train_df["text"]
    y_train = train_df["label"]
    
    if dev_df is not None and len(dev_df) >= min_dev_size:
        print(f"\nUsing provided dev set ({len(dev_df)} samples)")
        X_dev = dev_df["text"]
        y_dev = dev_df["label"]
    else:
        print(f"\nDev set too small or missing → splitting from train")
        try:
            X_train, X_dev, y_train, y_dev = train_test_split(
                X_train, y_train,
                test_size=test_size,
                random_state=random_state,
                stratify=y_train
            )
            print("  Using stratified split")
        except ValueError:
            print("  Stratified split failed → using non-stratified split")
            X_train, X_dev, y_train, y_dev = train_test_split(
                X_train, y_train,
                test_size=test_size,
                random_state=random_state
            )
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Dev:   {len(X_dev)}")
    
    return X_train, X_dev, y_train, y_dev


# ============================================================
# 5. Model Building
# ============================================================

def convert_ngram_range(params: dict) -> dict:
    """
    Convert ngram_range parameter from list to tuple.

    JSON configuration files cannot store Python tuples directly, so
    ngram_range is typically stored as a list [1, 3]. This function
    converts it to the tuple (1, 3) required by scikit-learn.

    Args:
        params (dict): Parameter dictionary that may contain 'ngram_range' key.

    Returns:
        dict: New dictionary with ngram_range converted to tuple if present.
            If ngram_range is not present or already a tuple, returns a copy
            of the original dictionary.

    Note:
        - Creates a copy to avoid modifying the original dictionary
        - Only converts if ngram_range exists and is a list
        - Handles other parameter types unchanged

    Example:
        >>> params = {"max_features": 5000, "ngram_range": [1, 3]}
        >>> converted = convert_ngram_range(params)
        >>> converted["ngram_range"]
        (1, 3)
        >>> type(converted["ngram_range"])
        <class 'tuple'>
    """
    if "ngram_range" in params and isinstance(params["ngram_range"], list):
        params = params.copy()
        params["ngram_range"] = tuple(params["ngram_range"])
    return params


def build_model(tfidf_params: dict = None, lr_params: dict = None):
    """
    Build TF-IDF + Logistic Regression classification pipeline.

    Creates a scikit-learn Pipeline combining TF-IDF vectorization and
    Logistic Regression classification. Uses sensible defaults if parameters
    are not provided.

    The pipeline consists of two steps:
        1. TF-IDF Vectorization: Converts text to numerical features
        2. Logistic Regression: Binary classification

    Args:
        tfidf_params (dict, optional): Parameters for TfidfVectorizer.
            Common parameters::

                {
                    "max_features": 5000,      # Maximum vocabulary size
                    "ngram_range": (1, 3),     # Use unigrams to trigrams
                    "min_df": 2,               # Minimum document frequency
                    "max_df": 0.95             # Maximum document frequency
                }

            Defaults to {"max_features": 5000, "ngram_range": (1, 3)}.

        lr_params (dict, optional): Parameters for LogisticRegression.
            Common parameters::

                {
                    "max_iter": 500,           # Maximum iterations
                    "class_weight": "balanced", # Handle class imbalance
                    "n_jobs": -1,              # Use all CPU cores
                    "C": 1.0                   # Regularization strength
                }

            Defaults to balanced model with 500 max iterations.

    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline with two steps:
            - "tfidf": TfidfVectorizer instance
            - "clf": LogisticRegression instance

    Note:
        - ngram_range in tfidf_params will be automatically converted from
          list to tuple if needed
        - class_weight="balanced" automatically adjusts for class imbalance
        - n_jobs=-1 uses all available CPU cores for faster training

    Example:
        >>> model = build_model()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)

        >>> custom_params = {
        ...     "max_features": 10000,
        ...     "ngram_range": [1, 2]
        ... }
        >>> model = build_model(tfidf_params=custom_params)
    """
    # Default TF-IDF parameters
    if tfidf_params is None:
        tfidf_params = {
            "max_features": 5000,
            "ngram_range": (1, 3)
        }
    else:
        # Convert ngram_range from list to tuple if needed
        tfidf_params = convert_ngram_range(tfidf_params)
    
    # Default Logistic Regression parameters
    if lr_params is None:
        lr_params = {
            "max_iter": 500,
            "class_weight": "balanced",
            "n_jobs": -1
        }
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LogisticRegression(**lr_params))
    ])
    
    return model


# ============================================================
# 6. Model Evaluation
# ============================================================

def evaluate_model(model, X, y, dataset_name: str = "Dataset"):
    """
    Evaluate trained model and print comprehensive metrics.

    Generates predictions and computes classification metrics including
    precision, recall, F1-score, and accuracy. Prints a detailed classification
    report.

    Args:
        model: Trained scikit-learn model or pipeline with predict() method.
        X (array-like): Input features (texts or feature vectors).
        y (array-like): True labels.
        dataset_name (str, optional): Name of dataset for display purposes.
            Defaults to "Dataset".

    Returns:
        tuple: Contains:
            - y_pred (np.ndarray): Predicted labels
            - acc (float): Accuracy score (0.0 to 1.0)

    Note:
        Prints classification report with:
            - Precision, Recall, F1-score for each class
            - Support (number of samples) for each class
            - Macro and weighted averages
            - Overall accuracy

    Example:
        >>> model = build_model()
        >>> model.fit(X_train, y_train)
        >>> y_pred, acc = evaluate_model(model, X_test, y_test, "Test Set")
        
        Test Set Results:
                      precision    recall  f1-score   support
        
                   0      0.850     0.920     0.884       100
                   1      0.900     0.810     0.852        90
        
            accuracy                          0.868       190
           macro avg      0.875     0.865     0.868       190
        weighted avg      0.873     0.868     0.869       190
        
        Test Set Accuracy: 0.868
    """
    y_pred = model.predict(X)
    
    print(f"\n{dataset_name} Results:")
    print(classification_report(y, y_pred, digits=3))
    acc = accuracy_score(y, y_pred)
    print(f"{dataset_name} Accuracy: {acc:.3f}")
    
    return y_pred, acc


# ============================================================
# 7. Rule-Based Prediction Enhancement
# ============================================================

def create_rule_based_predictor(model, pos_patterns: list, neg_patterns: list):
    """
    Create rule-based prediction wrapper for model enhancement.

    Generates a prediction function that applies hand-crafted rules to
    override model predictions. This is useful for incorporating domain
    knowledge and handling specific linguistic patterns.

    The rule-based approach works as follows:
        1. Model makes initial predictions
        2. Texts matching positive patterns → classified as Hope (1)
        3. Texts matching negative patterns → classified as Non-hope (0)
        4. Other texts keep model's prediction

    Args:
        model: Trained scikit-learn model with predict() method.
        pos_patterns (list): List of positive hope indicators (strings).
            Example: ["hope", "believe", "optimistic", "bright future"]
        neg_patterns (list): List of negative/non-hope indicators (strings).
            Example: ["no hope", "hopeless", "impossible", "never"]

    Returns:
        function: Prediction function with signature:
            predict_with_rules(texts) -> (base_preds, adjusted_preds)
            
            Where:
                - base_preds: Original model predictions
                - adjusted_preds: Predictions after applying rules

    Note:
        - Pattern matching is case-insensitive
        - Positive patterns take precedence over negative patterns
        - Patterns are substring matches (not exact word matches)
        - Empty pattern lists are allowed (no rules applied)

    Warning:
        Overly broad patterns may override correct model predictions.
        Test rule impacts on validation set before deployment.

    Example:
        >>> pos_patterns = ["hope", "optimistic"]
        >>> neg_patterns = ["no hope", "hopeless"]
        >>> predict_fn = create_rule_based_predictor(model, pos_patterns, neg_patterns)
        >>> 
        >>> texts = ["I have hope", "This is hopeless", "Normal text"]
        >>> base_preds, adjusted_preds = predict_fn(texts)
        >>> # "I have hope" → forced to 1
        >>> # "This is hopeless" → forced to 0
        >>> # "Normal text" → keeps model prediction
    """
    def contains_any(text: str, patterns: list) -> bool:
        """Check if text contains any pattern from the list."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in patterns)
    
    def predict_with_rules(texts):
        """
        Make predictions with rule-based overrides.

        Args:
            texts (array-like): Input texts to classify.

        Returns:
            tuple: (base_predictions, rule_adjusted_predictions)
        """
        base_preds = model.predict(texts)
        adjusted_preds = base_preds.copy()
        
        for i, text in enumerate(texts):
            # Positive patterns override to Hope_speech (1)
            if contains_any(text, pos_patterns):
                adjusted_preds[i] = 1
                continue
            # Negative patterns override to Non_hope_speech (0)
            if contains_any(text, neg_patterns):
                adjusted_preds[i] = 0
                continue
        
        return base_preds, adjusted_preds
    
    return predict_with_rules


# ============================================================
# 8. Language-Specific Training Pipeline
# ============================================================

def train_language_model(
    lang_name: str,
    lang_config: dict,
    base_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    balance_method: str = "undersample",
    tfidf_params: dict = None,
    lr_params: dict = None
):
    """
    Train complete model pipeline for a single language.

    Orchestrates the full training workflow including data loading, cleaning,
    balancing, model training, evaluation, and persistence. Optionally creates
    rule-enhanced variant if patterns are configured.

    This is the main training function that executes:
        1. Load train/dev/test data from processed CSVs
        2. Clean and filter data to binary labels
        3. Balance training data (under/over sampling)
        4. Prepare train/dev splits
        5. Build and train TF-IDF + LogReg model
        6. Evaluate on dev and test sets
        7. Save model to disk
        8. Create and save rule-based variant if patterns exist

    Args:
        lang_name (str): Language identifier ("english", "tamil", "malayalam").
        lang_config (dict): Language-specific configuration containing:
            - rule_patterns: Optional dict with "positive" and "negative" lists
        base_dir (Path): Project base directory.
        processed_dir (Path): Directory containing processed CSV files named:
            {lang_name}_train_processed.csv, etc.
        models_dir (Path): Directory where models will be saved.
        balance_method (str, optional): "undersample" or "oversample".
            Defaults to "undersample".
        tfidf_params (dict, optional): TF-IDF parameters. Defaults to
            max_features=5000, ngram_range=(1,3).
        lr_params (dict, optional): Logistic Regression parameters.
            Defaults to max_iter=500, class_weight="balanced".

    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline.

    Raises:
        FileNotFoundError: If required CSV files don't exist.
        Exception: Any exceptions during training are caught and logged.

    Note:
        Creates two model files:
            - hope_{lang}_model.pkl: Base model
            - hope_{lang}_model_with_rules.pkl: Model + rules (if patterns exist)

        Prints comprehensive progress information including:
            - Dataset sizes at each stage
            - Class distributions
            - Training progress
            - Evaluation metrics
            - File save confirmations

    Example:
        >>> from pathlib import Path
        >>> lang_config = {
        ...     "rule_patterns": {
        ...         "positive": ["hope", "optimistic"],
        ...         "negative": ["hopeless"]
        ...     }
        ... }
        >>> model = train_language_model(
        ...     lang_name="english",
        ...     lang_config=lang_config,
        ...     base_dir=Path("."),
        ...     processed_dir=Path("processed_data"),
        ...     models_dir=Path("models")
        ... )
        ============================================================
        Training ENGLISH Model
        ============================================================
        ...
        ✓ ENGLISH training complete!
    """
    print("\n" + "=" * 60)
    print(f"Training {lang_name.upper()} Model")
    print("=" * 60)
    
    # Construct file paths
    train_path = processed_dir / f"{lang_name}_train_processed.csv"
    dev_path = processed_dir / f"{lang_name}_dev_processed.csv"
    test_path = processed_dir / f"{lang_name}_test_processed.csv"
    
    print(f"\nLoading data:")
    print(f"  Train: {train_path}")
    print(f"  Dev:   {dev_path}")
    print(f"  Test:  {test_path}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path) if dev_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    
    print(f"\nLoaded:")
    print(f"  Train: {len(train_df)} rows")
    if dev_df is not None:
        print(f"  Dev:   {len(dev_df)} rows")
    if test_df is not None:
        print(f"  Test:  {len(test_df)} rows")
    
    # Clean data (keep only labels 0 and 1)
    train_df = clean_dataframe(train_df, keep_labels=[0, 1])
    if dev_df is not None:
        dev_df = clean_dataframe(dev_df, keep_labels=[0, 1])
    if test_df is not None:
        test_df = clean_dataframe(test_df, keep_labels=[0, 1])
    
    print(f"\nAfter cleaning (labels 0 & 1 only):")
    print(f"  Train: {len(train_df)} rows")
    if dev_df is not None:
        print(f"  Dev:   {len(dev_df)} rows")
    if test_df is not None:
        print(f"  Test:  {len(test_df)} rows")
    
    # Balance training data
    train_balanced = balance_train_data(train_df, method=balance_method)
    
    # Prepare train/dev split
    X_train, X_dev, y_train, y_dev = prepare_train_dev_split(
        train_balanced,
        dev_df
    )
    
    # Build model
    model = build_model(tfidf_params, lr_params)
    
    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Model trained!")
    
    # Evaluate on dev set
    if len(X_dev) > 0:
        evaluate_model(model, X_dev, y_dev, dataset_name="Validation")
    
    # Evaluate on test set
    if test_df is not None and len(test_df) > 0:
        X_test = test_df["text"]
        y_test = test_df["label"]
        evaluate_model(model, X_test, y_test, dataset_name="Test")
    
    # Save model
    model_path = models_dir / f"hope_{lang_name}_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Load rule patterns if available
    pos_patterns = lang_config.get("rule_patterns", {}).get("positive", [])
    neg_patterns = lang_config.get("rule_patterns", {}).get("negative", [])
    
    if pos_patterns or neg_patterns:
        print(f"\n✓ Rule-based patterns loaded:")
        print(f"  Positive patterns: {len(pos_patterns)}")
        print(f"  Negative patterns: {len(neg_patterns)}")
        
        # Create rule-based predictor
        predict_with_rules = create_rule_based_predictor(model, pos_patterns, neg_patterns)
        
        # Save rule-based predictor as well
        rule_model_path = models_dir / f"hope_{lang_name}_model_with_rules.pkl"
        joblib.dump({
            "model": model,
            "predict_with_rules": predict_with_rules,
            "pos_patterns": pos_patterns,
            "neg_patterns": neg_patterns
        }, rule_model_path)
        print(f"✓ Rule-based model saved to: {rule_model_path}")
    
    print(f"\n{'=' * 60}")
    print(f"✓ {lang_name.upper()} training complete!")
    print(f"{'=' * 60}")
    
    return model


# ============================================================
# 9. Main Orchestration
# ============================================================

def main():
    """
    Main training orchestrator - trains models for all configured languages.

    Executes the complete multi-language training pipeline:
        1. Loads configuration from config.json
        2. Sets up output directories
        3. Iterates through all configured languages
        4. Trains model for each language
        5. Saves all models
        6. Prints summary statistics

    The function is designed to be run as a standalone script and handles
    all file I/O, directory management, and error reporting.

    Configuration structure expected in config.json::

        {
            "processed_dir": "processed_data",
            "models_dir": "models",
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
            },
            "languages": {
                "english": {...},
                "tamil": {...},
                "malayalam": {...}
            }
        }

    Raises:
        FileNotFoundError: If config.json is not found.
        KeyError: If required configuration keys are missing.

    Note:
        - Creates models_dir if it doesn't exist
        - Continues training other languages if one fails
        - Prints comprehensive progress and summary information
        - Each language model is independent (failure doesn't stop others)

    Example:
        Expected console output::

            ============================================================
            HOPE SPEECH BASELINE MODEL TRAINING
            ============================================================
            
            Configuration:
              Processed data dir: processed_data
              Models output dir:  models
              Balance method:     undersample
              TF-IDF params:      {...}
              LogReg params:      {...}
            
            ============================================================
            Training ENGLISH Model
            ============================================================
            ...
            
            ============================================================
            TRAINING SUMMARY
            ============================================================
            
            Successfully trained models for 3 languages:
              ✓ English: models/hope_english_model.pkl
              ✓ Tamil: models/hope_tamil_model.pkl
              ✓ Malayalam: models/hope_malayalam_model.pkl
            
            ============================================================
            ✓ ALL TRAINING COMPLETE!
            ============================================================

    See Also:
        train_language_model: Individual language training function
        load_config: Configuration loading function
    """
    # Load configuration
    base_dir = Path.cwd()
    config = load_config(base_dir / "config.json")
    
    # Setup directories
    processed_dir = base_dir / config["processed_dir"]
    models_dir = base_dir / config.get("models_dir", "models")
    models_dir.mkdir(exist_ok=True)
    
    # Training parameters
    balance_method = config.get("training", {}).get("balance_method", "undersample")
    
    tfidf_params = config.get("training", {}).get("tfidf_params", {
        "max_features": 5000,
        "ngram_range": (1, 3)
    })
    
    lr_params = config.get("training", {}).get("lr_params", {
        "max_iter": 500,
        "class_weight": "balanced",
        "n_jobs": -1
    })
    
    print("\n" + "=" * 60)
    print("HOPE SPEECH BASELINE MODEL TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Processed data dir: {processed_dir}")
    print(f"  Models output dir:  {models_dir}")
    print(f"  Balance method:     {balance_method}")
    print(f"  TF-IDF params:      {tfidf_params}")
    print(f"  LogReg params:      {lr_params}")
    
    # Train models for all languages
    trained_models = {}
    
    for lang_name, lang_config in config["languages"].items():
        try:
            model = train_language_model(
                lang_name=lang_name,
                lang_config=lang_config,
                base_dir=base_dir,
                processed_dir=processed_dir,
                models_dir=models_dir,
                balance_method=balance_method,
                tfidf_params=tfidf_params,
                lr_params=lr_params
            )
            trained_models[lang_name] = model
        except Exception as e:
            print(f"\n❌ Error training {lang_name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\nSuccessfully trained models for {len(trained_models)} languages:")
    for lang_name in trained_models.keys():
        model_path = models_dir / f"hope_{lang_name}_model.pkl"
        print(f"  ✓ {lang_name.capitalize()}: {model_path}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()