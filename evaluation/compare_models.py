"""
Model comparison — baseline TF-IDF vs XLM-R vs MuRIL (test set).

Reads from metrics/:
    - baseline_metrics.json            (written by training/train_all.py)
    - xlmr_{lang}_metrics.json         (downloaded from Colab / Drive)
    - muril_{lang}_metrics.json        (downloaded from Colab / Drive)

Prints a per-language comparison table and writes:
    - metrics/model_comparison.md      (markdown, paste into README/report)
    - metrics/model_comparison.csv     (for plots/spreadsheets)

Usage (from anywhere):
    $ python evaluation/compare_models.py
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "metrics"

LANGUAGES = ["english", "tamil", "malayalam"]
METRIC_KEYS = ["accuracy", "macro_f1", "weighted_f1",
               "hope_f1", "hope_precision", "hope_recall"]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect() -> dict:
    """Gather test metrics for every (language, model) that exists on disk."""
    rows = {}   # (lang, model_label) -> metrics dict

    baseline = load_json(METRICS_DIR / "baseline_metrics.json")
    if baseline:
        for lang in LANGUAGES:
            test = baseline.get(lang, {}).get("test", {})
            if "base" in test:
                rows[(lang, "TF-IDF + LR")] = test["base"]
            if "with_rules" in test:
                rows[(lang, "TF-IDF + LR + rules")] = test["with_rules"]

    for lang in LANGUAGES:
        for prefix, label in [("xlmr", "XLM-R"), ("muril", "MuRIL")]:
            payload = load_json(METRICS_DIR / f"{prefix}_{lang}_metrics.json")
            if payload and "test" in payload:
                rows[(lang, label)] = payload["test"]
    return rows


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def main():
    rows = collect()
    if not rows:
        raise SystemExit(f"No metric files found in {METRICS_DIR}")

    header = ["Language", "Model"] + [k.replace("_", " ").title()
                                      for k in METRIC_KEYS]
    table = []
    for lang in LANGUAGES:
        lang_rows = [(m, v) for (l, m), v in rows.items() if l == lang]
        if not lang_rows:
            continue
        # best macro F1 per language gets a marker
        best = max(v.get("macro_f1", 0) for _, v in lang_rows)
        for model, v in lang_rows:
            star = " *" if v.get("macro_f1", 0) == best else ""
            table.append([lang.capitalize(), model + star]
                         + [fmt(v.get(k)) for k in METRIC_KEYS])

    # --- print ---
    widths = [max(len(str(r[i])) for r in [header] + table)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    prev_lang = None
    for r in table:
        if prev_lang is not None and r[0] != prev_lang:
            print()
        prev_lang = r[0]
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print("\n(* = best test macro F1 for that language)")

    # --- markdown ---
    md_path = METRICS_DIR / "model_comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Model comparison (test set)\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in table:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        f.write("\n`*` = best test macro F1 for that language\n")
    print(f"\nMarkdown table -> {md_path}")

    # --- csv ---
    csv_path = METRICS_DIR / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f"CSV            -> {csv_path}")


if __name__ == "__main__":
    main()
