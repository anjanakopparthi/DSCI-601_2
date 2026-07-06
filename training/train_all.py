"""
Multilingual Hope Speech Detection - Baseline Training (v2, binary).

Trains a TF-IDF + Logistic Regression baseline per language and evaluates a
rule-enhanced variant, on the binary processed data (0 = Non-hope, 1 = Hope).

Fixes over the previous version:
    1. No undersampling. The full training set is used; class imbalance is
       handled by LogisticRegression(class_weight="balanced"). The old script
       both undersampled AND used balanced weights (redundant), throwing away
       ~90% of English majority-class rows.
    2. The rules variant is saved as a plain dict {model, pos_patterns,
       neg_patterns} — the old version pickled a closure, which is not
       serializable and produced corrupt files. Use load_rules_model() +
       predict_with_rules() from this module to run it.
    3. The rules variant is actually EVALUATED on dev and test (previously it
       was saved but never scored).
    4. Reports accuracy, macro F1, weighted F1, and Hope-class F1 (macro and
       Hope F1 are the honest numbers on imbalanced data).
    5. All metrics are saved to metrics/baseline_metrics.json for later
       comparison against XLM-R.
    6. random_state everywhere for reproducibility.

Usage (from repo root):
    $ python training/train_all.py

Reads config.json from the current directory. Expects processed CSVs in
`processed/`, writes models to `models/` and metrics to `metrics/`.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Configuration
# ============================================================

def load_config(config_path: Path) -> dict:
    """Load config.json (must exist)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. Data loading
# ============================================================

def load_split(processed_dir: Path, lang: str, split: str) -> pd.DataFrame:
    """Load one processed split and sanity-check it is clean and binary."""
    path = processed_dir / f"{lang}_{split}_processed.csv"
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    assert set(df["label"].unique()) <= {0, 1}, (
        f"{path} contains non-binary labels — rerun preprocessing"
    )
    return df.reset_index(drop=True)


# ============================================================
# 3. Model building
# ============================================================

def build_model(tfidf_params: dict, lr_params: dict) -> Pipeline:
    """TF-IDF + Logistic Regression pipeline (params come from config)."""
    tfidf_params = dict(tfidf_params)
    if isinstance(tfidf_params.get("ngram_range"), list):
        tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
    return Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LogisticRegression(**lr_params)),
    ])


# ============================================================
# 4. Rule-based prediction (module-level, picklable-friendly)
# ============================================================

def predict_with_rules(model, texts, pos_patterns, neg_patterns):
    """
    Model predictions with substring-rule overrides.

    - text contains any positive pattern  -> forced to 1 (Hope)
    - else text contains any neg pattern  -> forced to 0 (Non-hope)
    - otherwise                            -> model prediction

    Matching is case-insensitive substring. Positive rules take precedence.
    """
    base = model.predict(texts)
    adjusted = base.copy()
    pos = [p.lower() for p in pos_patterns]
    neg = [n.lower() for n in neg_patterns]
    for i, text in enumerate(texts):
        t = str(text).lower()
        if any(p in t for p in pos):
            adjusted[i] = 1
        elif any(n in t for n in neg):
            adjusted[i] = 0
    return adjusted


def load_rules_model(path):
    """
    Load a rules-model file saved by this script.

    Returns (model, pos_patterns, neg_patterns). Predict with:
        preds = predict_with_rules(model, texts, pos_patterns, neg_patterns)
    """
    bundle = joblib.load(path)
    return bundle["model"], bundle["pos_patterns"], bundle["neg_patterns"]


# ============================================================
# 5. Evaluation
# ============================================================

def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy + macro/weighted/Hope-class precision-recall-F1 + confusion."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "hope_f1": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "hope_precision": round(precision_score(y_true, y_pred, pos_label=1,
                                                zero_division=0), 4),
        "hope_recall": round(recall_score(y_true, y_pred, pos_label=1,
                                          zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate(model, df: pd.DataFrame, name: str,
             rules: tuple | None = None) -> dict:
    """Evaluate base model (and rules variant if patterns given) on a split."""
    X, y = df["text"], df["label"]
    y_pred = model.predict(X)
    out = {"base": compute_metrics(y, y_pred)}

    print(f"\n--- {name} (base model) ---")
    print(classification_report(y, y_pred, digits=3,
                                target_names=["Non-hope (0)", "Hope (1)"]))

    if rules is not None:
        pos_patterns, neg_patterns = rules
        y_rules = predict_with_rules(model, list(X), pos_patterns, neg_patterns)
        out["with_rules"] = compute_metrics(y, y_rules)
        n_changed = int((y_rules != y_pred).sum())
        out["with_rules"]["n_predictions_changed_by_rules"] = n_changed
        print(f"--- {name} (with rules: {n_changed} predictions changed) ---")
        print(f"  macro F1 {out['base']['macro_f1']:.4f} -> "
              f"{out['with_rules']['macro_f1']:.4f} | "
              f"Hope F1 {out['base']['hope_f1']:.4f} -> "
              f"{out['with_rules']['hope_f1']:.4f}")
    return out


# ============================================================
# 6. Per-language pipeline
# ============================================================

def train_language(lang: str, lang_cfg: dict, processed_dir: Path,
                   models_dir: Path, tfidf_params: dict,
                   lr_params: dict) -> dict:
    """Train, evaluate, and save base + rules models for one language."""
    print("\n" + "=" * 60)
    print(f"Training {lang.upper()} baseline")
    print("=" * 60)

    train_df = load_split(processed_dir, lang, "train")
    dev_df = load_split(processed_dir, lang, "dev")
    test_df = load_split(processed_dir, lang, "test")
    print(f"Train: {len(train_df)}  (Hope: {int(train_df['label'].sum())}, "
          f"Non-hope: {int((train_df['label'] == 0).sum())})")
    print(f"Dev:   {len(dev_df)} | Test: {len(test_df)}")

    model = build_model(tfidf_params, lr_params)
    model.fit(train_df["text"], train_df["label"])

    pos_patterns = lang_cfg.get("rule_patterns", {}).get("positive", [])
    neg_patterns = lang_cfg.get("rule_patterns", {}).get("negative", [])
    rules = (pos_patterns, neg_patterns) if (pos_patterns or neg_patterns) else None

    results = {
        "dev": evaluate(model, dev_df, f"{lang} DEV", rules),
        "test": evaluate(model, test_df, f"{lang} TEST", rules),
    }

    # --- save base model ---
    base_path = models_dir / f"hope_{lang}_model.pkl"
    joblib.dump(model, base_path)
    print(f"\nSaved base model  -> {base_path}")

    # --- save rules bundle (plain dict: no closures, loads cleanly) ---
    if rules is not None:
        rules_path = models_dir / f"hope_{lang}_model_with_rules.pkl"
        joblib.dump({
            "model": model,
            "pos_patterns": pos_patterns,
            "neg_patterns": neg_patterns,
        }, rules_path)
        print(f"Saved rules model -> {rules_path}")

    return results


# ============================================================
# 7. Entry point
# ============================================================

def main():
    # Repo root = one level above the folder this script lives in.
    # Works from any working directory (terminal, VS Code debugger, etc.)
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config(base_dir / "config.json")

    processed_dir = base_dir / config["processed_dir"]
    models_dir = base_dir / config.get("models_dir", "models")
    metrics_dir = base_dir / "metrics"
    models_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    tfidf_params = config["training"]["tfidf_params"]
    lr_params = config["training"]["lr_params"]

    print("=" * 60)
    print("HOPE SPEECH BASELINE TRAINING (binary, full data + class weights)")
    print("=" * 60)
    print(f"TF-IDF: {tfidf_params}")
    print(f"LogReg: {lr_params}")

    all_results = {}
    for lang, lang_cfg in config["languages"].items():
        all_results[lang] = train_language(
            lang, lang_cfg, processed_dir, models_dir, tfidf_params, lr_params
        )

    metrics_path = metrics_dir / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved -> {metrics_path}")

    print("\n" + "=" * 60)
    print("SUMMARY (test set)")
    print("=" * 60)
    print(f"{'Language':<12}{'Acc':>8}{'MacroF1':>10}{'HopeF1':>10}"
          f"{'MacroF1+rules':>16}")
    for lang, res in all_results.items():
        t = res["test"]
        rules_f1 = t.get("with_rules", {}).get("macro_f1", "-")
        print(f"{lang.capitalize():<12}{t['base']['accuracy']:>8.3f}"
              f"{t['base']['macro_f1']:>10.3f}{t['base']['hope_f1']:>10.3f}"
              f"{rules_f1 if isinstance(rules_f1, str) else format(rules_f1, '>16.3f')}")


if __name__ == "__main__":
    main()