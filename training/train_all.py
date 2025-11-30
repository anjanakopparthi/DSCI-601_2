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
# 1. Load configuration
# ============================================================

def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Data cleaning and filtering
# ============================================================

def clean_dataframe(df: pd.DataFrame, keep_labels: list = [0, 1]) -> pd.DataFrame:
    """
    Clean dataframe by:
    - Keeping only specified labels
    - Removing empty text
    - Converting text to string
    """
    df = df.copy()
    df = df[df["label"].isin(keep_labels)]
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    return df


# ============================================================
# 3. Balance training data
# ============================================================

def balance_train_data(df: pd.DataFrame, method: str = "undersample", random_state: int = 42) -> pd.DataFrame:
    """
    Balance training data using undersample or oversample
    
    Args:
        df: DataFrame with 'label' column
        method: 'undersample' or 'oversample'
        random_state: Random seed
    
    Returns:
        Balanced DataFrame
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
# 4. Create validation split if needed
# ============================================================

def prepare_train_dev_split(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame = None,
    test_size: float = 0.2,
    random_state: int = 42,
    min_dev_size: int = 5
):
    """
    Prepare train/dev split:
    - Use provided dev set if it exists and is large enough
    - Otherwise split from training data
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
# 5. Build and train model
# ============================================================

def convert_ngram_range(params: dict) -> dict:
    """Convert ngram_range from list to tuple (JSON doesn't support tuples)"""
    if "ngram_range" in params and isinstance(params["ngram_range"], list):
        params = params.copy()
        params["ngram_range"] = tuple(params["ngram_range"])
    return params


def build_model(tfidf_params: dict = None, lr_params: dict = None):
    """Build TF-IDF + Logistic Regression pipeline"""
    
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
# 6. Evaluate model
# ============================================================

def evaluate_model(model, X, y, dataset_name: str = "Dataset"):
    """Evaluate model and print results"""
    y_pred = model.predict(X)
    
    print(f"\n{dataset_name} Results:")
    print(classification_report(y, y_pred, digits=3))
    acc = accuracy_score(y, y_pred)
    print(f"{dataset_name} Accuracy: {acc:.3f}")
    
    return y_pred, acc


# ============================================================
# 7. Rule-based prediction wrapper
# ============================================================

def create_rule_based_predictor(model, pos_patterns: list, neg_patterns: list):
    """
    Create a rule-based prediction function that overrides model predictions
    
    Args:
        model: Trained sklearn pipeline
        pos_patterns: List of positive hope patterns
        neg_patterns: List of negative/no-hope patterns
    
    Returns:
        Function that takes texts and returns (base_preds, rule_adjusted_preds)
    """
    
    def contains_any(text: str, patterns: list) -> bool:
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in patterns)
    
    def predict_with_rules(texts):
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
# 8. Main training function for one language
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
    Train model for one language
    
    Args:
        lang_name: Language name (e.g., 'english', 'tamil', 'malayalam')
        lang_config: Language-specific config from config.json
        base_dir: Base directory path
        processed_dir: Directory with processed CSV files
        models_dir: Directory to save models
        balance_method: 'undersample' or 'oversample'
        tfidf_params: TF-IDF parameters (optional)
        lr_params: Logistic Regression parameters (optional)
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
# 9. Main orchestrator
# ============================================================

def main():
    """Main training orchestrator - trains all languages from config"""
    
    # Load configuration
    base_dir = Path.cwd()
    config = load_config(base_dir / "config.json")
    
    # Setup directories
    processed_dir = base_dir / config["processed_dir"]
    models_dir = base_dir / config.get("models_dir", "models")  # Read from config, default to "models"
    models_dir.mkdir(exist_ok=True)
    
    # Training parameters (can be added to config.json if needed)
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
