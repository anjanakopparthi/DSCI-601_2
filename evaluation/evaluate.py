"""
Multilingual Hope Speech Detection - Unified Model Evaluation.

Replaces the three per-language scripts (forEnglish.py / forMalayalam.py /
forTamil.py) with a single parameterized evaluator. Works with the binary
models produced by training/train_all.py and the processed data produced by
preprocess/preprocess_all.py.

For each language it evaluates:
    - the base model            (models/hope_{lang}_model.pkl)
    - the rules-enhanced model  (models/hope_{lang}_model_with_rules.pkl)

and reports accuracy, macro F1, weighted F1, Hope-class precision/recall/F1,
the full classification report, and the confusion matrix. Results are saved
to metrics/evaluation_{split}.json.

Usage (from repo root):
    $ python evaluation/evaluate.py                     # all languages, test
    $ python evaluation/evaluate.py --lang tamil        # one language
    $ python evaluation/evaluate.py --split dev         # dev instead of test
    $ python evaluation/evaluate.py --lang english --split dev
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

LANGUAGES = ["english", "tamil", "malayalam"]


# ============================================================
# 1. Loading
# ============================================================

def load_split(processed_dir: Path, lang: str, split: str) -> pd.DataFrame:
    """Load one processed split; enforce clean binary data."""
    path = processed_dir / f"{lang}_{split}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run preprocess/preprocess_all.py first"
        )
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    assert set(df["label"].unique()) <= {0, 1}, f"non-binary labels in {path}"
    return df.reset_index(drop=True)


def predict_with_rules(model, texts, pos_patterns, neg_patterns):
    """Rule-override predictions (same logic as training/train_all.py)."""
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


# ============================================================
# 2. Metrics
# ============================================================

def compute_metrics(y_true, y_pred) -> dict:
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


def print_block(y_true, y_pred, title: str):
    print(f"\n--- {title} ---")
    print(classification_report(y_true, y_pred, digits=3,
                                target_names=["Non-hope (0)", "Hope (1)"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred))


# ============================================================
# 3. Per-language evaluation
# ============================================================

def evaluate_language(lang: str, split: str, processed_dir: Path,
                      models_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print(f"Evaluating {lang.upper()} on {split.upper()}")
    print("=" * 60)

    df = load_split(processed_dir, lang, split)
    X, y = df["text"], df["label"]
    print(f"Samples: {len(df)}  (Hope: {int(y.sum())}, "
          f"Non-hope: {int((y == 0).sum())})")

    results = {}

    # --- base model ---
    base_path = models_dir / f"hope_{lang}_model.pkl"
    model = joblib.load(base_path)
    y_pred = model.predict(X)
    results["base"] = compute_metrics(y, y_pred)
    print_block(y, y_pred, f"{lang} {split} — base model")

    # --- rules model (optional file) ---
    rules_path = models_dir / f"hope_{lang}_model_with_rules.pkl"
    if rules_path.exists():
        bundle = joblib.load(rules_path)
        y_rules = predict_with_rules(bundle["model"], list(X),
                                     bundle["pos_patterns"],
                                     bundle["neg_patterns"])
        results["with_rules"] = compute_metrics(y, y_rules)
        results["with_rules"]["n_predictions_changed_by_rules"] = \
            int((y_rules != y_pred).sum())
        print_block(y, y_rules, f"{lang} {split} — with rules")
    else:
        print(f"(no rules model at {rules_path}, skipping)")

    return results


# ============================================================
# 4. Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate hope speech models")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()

    base_dir = Path.cwd()
    config_path = base_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    processed_dir = base_dir / config["processed_dir"]
    models_dir = base_dir / config.get("models_dir", "models")
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    all_results = {
        lang: evaluate_language(lang, args.split, processed_dir, models_dir)
        for lang in langs
    }

    out_path = metrics_dir / f"evaluation_{args.split}.json"
    # merge with existing file so single-language runs don't clobber others
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
    else:
        merged = {}
    merged.update(all_results)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    print("\n" + "=" * 60)
    print(f"SUMMARY ({args.split} set)")
    print("=" * 60)
    print(f"{'Language':<12}{'Variant':<12}{'Acc':>8}{'MacroF1':>10}"
          f"{'WeightedF1':>12}{'HopeF1':>10}")
    for lang, res in all_results.items():
        for variant, m in res.items():
            print(f"{lang.capitalize():<12}{variant:<12}{m['accuracy']:>8.3f}"
                  f"{m['macro_f1']:>10.3f}{m['weighted_f1']:>12.3f}"
                  f"{m['hope_f1']:>10.3f}")


if __name__ == "__main__":
    main()