"""
Gold-vs-silver cross-evaluation report.

Aggregates all cross-evaluation results into per-language 2x2 matrices for
both model families:

                         | gold test | silver test
    ---------------------+-----------+-------------
    gold-trained model   |    ...    |    ...
    silver-trained model |    ...    |    ...

Sources (in metrics/):
    silver_baseline_metrics.json        TF-IDF matrix (train_silver_baseline.py)
    xlmr_{lang}_metrics.json            gold-trained XLM-R -> gold test
    xlmr_silver_{lang}_metrics.json     the other three XLM-R cells

Outputs:
    prints the matrices; writes metrics/cross_evaluation_report.md

Usage (from anywhere):
    $ python evaluation/cross_eval_report.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "metrics"
LANGUAGES = ["english", "tamil", "malayalam"]
CELLS = ["gold_trained__gold_test", "gold_trained__silver_test",
         "silver_trained__gold_test", "silver_trained__silver_test"]


def load_json(name):
    path = METRICS_DIR / name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect() -> dict:
    """-> {lang: {family: {cell: metrics}}}"""
    out = {lang: {"TF-IDF + LR": {}, "XLM-R": {}} for lang in LANGUAGES}

    baseline = load_json("silver_baseline_metrics.json") or {}
    for lang in LANGUAGES:
        for cell, m in baseline.get(lang, {}).items():
            out[lang]["TF-IDF + LR"][cell] = m

    for lang in LANGUAGES:
        gold = load_json(f"xlmr_{lang}_metrics.json")
        if gold and "test" in gold:
            out[lang]["XLM-R"]["gold_trained__gold_test"] = gold["test"]
        silver = load_json(f"xlmr_silver_{lang}_metrics.json")
        if silver:
            for cell, m in silver.get("cross_evaluation", {}).items():
                out[lang]["XLM-R"][cell] = m
    return out


def fmt(m, key="macro_f1"):
    return f"{m[key]:.3f}" if m and key in m else "  -  "


def matrix_lines(cells: dict) -> list[str]:
    return [
        f"{'':>16}{'gold test':>12}{'silver test':>13}",
        f"{'gold-trained':>16}"
        f"{fmt(cells.get('gold_trained__gold_test')):>12}"
        f"{fmt(cells.get('gold_trained__silver_test')):>13}",
        f"{'silver-trained':>16}"
        f"{fmt(cells.get('silver_trained__gold_test')):>12}"
        f"{fmt(cells.get('silver_trained__silver_test')):>13}",
    ]


def main():
    data = collect()
    md = ["# Cross-evaluation report (macro F1)", "",
          "Rows = training labels, columns = evaluation labels. "
          "On-diagonal cells measure fit to a label standard; off-diagonal "
          "cells measure transfer between the gold (human) and silver (LLM) "
          "standards.", ""]

    for lang in LANGUAGES:
        print("\n" + "=" * 46)
        print(f"{lang.upper()}")
        print("=" * 46)
        md.append(f"## {lang.capitalize()}\n")
        for family, cells in data[lang].items():
            if not cells:
                continue
            print(f"\n{family} (macro F1)")
            for line in matrix_lines(cells):
                print(line)
            md.append(f"**{family}**\n")
            md.append("|  | gold test | silver test |")
            md.append("|---|---|---|")
            md.append(f"| gold-trained | "
                      f"{fmt(cells.get('gold_trained__gold_test'))} | "
                      f"{fmt(cells.get('gold_trained__silver_test'))} |")
            md.append(f"| silver-trained | "
                      f"{fmt(cells.get('silver_trained__gold_test'))} | "
                      f"{fmt(cells.get('silver_trained__silver_test'))} |")
            md.append("")

    out = METRICS_DIR / "cross_evaluation_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown report -> {out}")


if __name__ == "__main__":
    main()
