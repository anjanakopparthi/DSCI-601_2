"""
Build silver-standard train/dev/test splits from the LLM-labeled corpora.

Input : ``data_collection/collected/{lang}_corpus_labeled.csv``

Output: ``processed_silver/{lang}_{split}_silver.csv``
(columns: text, label_str, label — same format as processed/, so all
existing training/eval code works by just pointing at this folder)

Steps per language:
    1. Drop failed labels (label == -1) and exact-duplicate texts.
    2. CONTAMINATION CONTROL: drop any comment whose normalized text appears
       in the GOLD dev or test split — silver-trained models will later be
       evaluated on gold test, so silver training data must not contain it.
    3. Stratified 80/10/10 train/dev/test split (seed 42), preserving the
       corpus's natural label distribution.

Usage (from anywhere):
    $ python data_collection/build_silver_datasets.py
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
COLLECTED = Path(__file__).resolve().parent / "collected"
OUT_DIR = BASE_DIR / "processed_silver"

LANGUAGES = ["english", "tamil", "malayalam"]
SEED = 42
WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    """Normalization for overlap matching (case/whitespace-insensitive)."""
    return WS_RE.sub(" ", str(text)).strip().lower()


def build_language(lang: str) -> dict:
    labeled_path = COLLECTED / f"{lang}_corpus_labeled.csv"
    df = pd.read_csv(labeled_path)
    n0 = len(df)
    df["text"] = df["text"].astype(str)
    df = df[df["label"].isin([0, 1])]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # --- contamination control against gold dev/test ---
    gold_eval_texts = set()
    for split in ("dev", "test"):
        gold = pd.read_csv(BASE_DIR / "processed" /
                           f"{lang}_{split}_processed.csv")
        gold_eval_texts.update(norm(t) for t in gold["text"].astype(str))
    before = len(df)
    df = df[~df["text"].map(norm).isin(gold_eval_texts)]
    n_contaminated = before - len(df)

    df["label_str"] = df["label"].map({1: "Hope_speech", 0: "Non_hope_speech"})
    df = df[["text", "label_str", "label"]]

    # --- stratified 80/10/10 ---
    train, rest = train_test_split(df, test_size=0.20, random_state=SEED,
                                   stratify=df["label"])
    dev, test = train_test_split(rest, test_size=0.50, random_state=SEED,
                                 stratify=rest["label"])

    OUT_DIR.mkdir(exist_ok=True)
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        part.to_csv(OUT_DIR / f"{lang}_{name}_silver.csv", index=False)

    stats = {"raw": n0, "usable": len(df), "contaminated": n_contaminated,
             "train": len(train), "dev": len(dev), "test": len(test),
             "hope_pct": df["label"].mean()}
    print(f"[{lang}] raw {n0} -> usable {len(df)} "
          f"(dropped {n_contaminated} gold-eval overlaps)")
    print(f"  train {len(train)} / dev {len(dev)} / test {len(test)}  "
          f"(hope: {stats['hope_pct']:.1%})")
    return stats


def main():
    print("Building silver datasets (contamination-controlled, "
          "stratified 80/10/10)\n")
    all_stats = {lang: build_language(lang) for lang in LANGUAGES}
    print("\nSUMMARY")
    print(f"{'Language':<12}{'Train':>8}{'Dev':>7}{'Test':>7}{'Hope%':>8}")
    for lang, s in all_stats.items():
        print(f"{lang.capitalize():<12}{s['train']:>8}{s['dev']:>7}"
              f"{s['test']:>7}{s['hope_pct']:>8.1%}")
    print(f"\nSaved -> {OUT_DIR}")
    print("NOTE: silver data is derived from scraped user comments — "
          "processed_silver/ should stay gitignored like processed/.")


if __name__ == "__main__":
    main()
