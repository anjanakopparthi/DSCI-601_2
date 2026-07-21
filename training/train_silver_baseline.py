"""
Silver-track baseline: train TF-IDF + LR on the LLM-labeled silver data and
cross-evaluate against both test sets.

For each language this script produces the full baseline half of the
cross-evaluation matrix:

                          | silver test | gold test
    ----------------------+-------------+-----------
    gold-trained TF-IDF   |     X       |   (already known)
    silver-trained TF-IDF |     X       |     X

The gold-trained baseline is loaded from models/hope_{lang}_model.pkl
(trained by training/train_all.py); the silver-trained one is trained here
with identical hyperparameters (from config.json) and saved to
models/hope_{lang}_model_silver.pkl.

Results -> metrics/silver_baseline_metrics.json

Usage (from anywhere):
    $ python training/train_silver_baseline.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "training"))

from train_all import build_model, compute_metrics, load_config  # noqa: E402

LANGUAGES = ["english", "tamil", "malayalam"]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    assert set(df["label"].unique()) <= {0, 1}, path
    return df.reset_index(drop=True)


def main():
    config = load_config(BASE_DIR / "config.json")
    tfidf_params = config["training"]["tfidf_params"]
    lr_params = config["training"]["lr_params"]
    models_dir = BASE_DIR / "models"
    metrics_dir = BASE_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    results = {}
    for lang in LANGUAGES:
        print("\n" + "=" * 60)
        print(f"{lang.upper()}: silver baseline + cross-evaluation")
        print("=" * 60)

        silver_train = load_csv(BASE_DIR / "processed_silver" /
                                f"{lang}_train_silver.csv")
        silver_test = load_csv(BASE_DIR / "processed_silver" /
                               f"{lang}_test_silver.csv")
        gold_test = load_csv(BASE_DIR / "processed" /
                             f"{lang}_test_processed.csv")
        print(f"silver train {len(silver_train)} "
              f"(hope {silver_train['label'].mean():.1%}) | "
              f"silver test {len(silver_test)} | gold test {len(gold_test)}")

        # --- train silver baseline (identical recipe to gold baseline) ---
        model_silver = build_model(tfidf_params, lr_params)
        model_silver.fit(silver_train["text"], silver_train["label"])
        silver_model_path = models_dir / f"hope_{lang}_model_silver.pkl"
        joblib.dump(model_silver, silver_model_path)

        # --- load gold-trained baseline ---
        gold_model_path = models_dir / f"hope_{lang}_model.pkl"
        if not gold_model_path.exists():
            sys.exit(f"{gold_model_path} missing — run training/train_all.py")
        model_gold = joblib.load(gold_model_path)

        # --- 2x2 cross-evaluation ---
        cell = {}
        for train_tag, model in (("gold_trained", model_gold),
                                 ("silver_trained", model_silver)):
            for test_tag, test_df in (("gold_test", gold_test),
                                      ("silver_test", silver_test)):
                m = compute_metrics(test_df["label"],
                                    model.predict(test_df["text"]))
                cell[f"{train_tag}__{test_tag}"] = m
                print(f"  {train_tag:>14} -> {test_tag:<11}: "
                      f"acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  "
                      f"hopeF1 {m['hope_f1']:.3f}")
        results[lang] = cell
        print(f"  saved silver model -> {silver_model_path}")

    out = metrics_dir / "silver_baseline_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll cross-eval metrics saved -> {out}")

    print("\nSUMMARY (macro F1)")
    print(f"{'Language':<12}{'gold->gold':>11}{'gold->silv':>11}"
          f"{'silv->silv':>11}{'silv->gold':>11}")
    for lang, c in results.items():
        print(f"{lang.capitalize():<12}"
              f"{c['gold_trained__gold_test']['macro_f1']:>11.3f}"
              f"{c['gold_trained__silver_test']['macro_f1']:>11.3f}"
              f"{c['silver_trained__silver_test']['macro_f1']:>11.3f}"
              f"{c['silver_trained__gold_test']['macro_f1']:>11.3f}")


if __name__ == "__main__":
    main()
