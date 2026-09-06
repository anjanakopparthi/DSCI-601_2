# Multilingual Hope Speech Detection

**Hope speech detection for English, Tamil, and Malayalam — and an investigation of the benchmark's label quality with an LLM-labeled silver-standard re-annotation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Binary hope speech classification (`0 = Non_hope_speech`, `1 = Hope_speech`) on
the HopeEDI dataset (LT-EDI shared task), covering native script, romanized,
and code-mixed text. DSCI-601 capstone project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start — Gold Track](#quick-start--gold-track)
- [Silver Track — Data Collection & LLM Labeling](#silver-track--data-collection--llm-labeling)
- [Transformer Fine-tuning (Colab)](#transformer-fine-tuning-colab)
- [Testing](#testing)
- [Documentation](#documentation)
- [Ethics & Data Notes](#ethics--data-notes)
- [License](#license)
- [Citation](#citation)

---

## Project Overview

The project has three phases:

**Phase 1 — Gold pipeline.** A rebuilt binary preprocessing / training /
evaluation pipeline (not-in-language rows removed, transformer-friendly
cleaning, deduplication, 92-test suite), a TF-IDF + Logistic Regression
baseline, then fine-tuned **XLM-R**, **MuRIL**, and **MuRIL + domain-adaptive
pretraining (DAPT)** on 12k scraped in-domain Tamil comments.

**Phase 2 — Label-quality investigation.** LLM labeling was validated against
gold *before* use. Agreement tracked each subset's known annotation quality —
Malayalam **0.855**, English **0.782** (few-shot), Tamil **0.545** (κ = 0.09) —
and a manual audit of Tamil disagreements found gold `Hope_speech` labels on
insults, jokes, and neutral questions, explaining the ~0.63 macro-F1 ceiling
that every gold-trained model converges to.

**Phase 3 — Silver track.** 46,875 topic-matched YouTube comments collected
via the official Data API (anonymized at collection), filtered per language
(script detection + romanized-Tamil / Manglish lexicons), labeled with the
validated setup (`gemini-3.1-flash-lite` + 30 gold few-shot examples;
**90/100 human-verified accuracy** on an audited random English sample),
split with gold-contamination control, and used to retrain both model
families for a full gold-vs-silver cross-evaluation.

---

## Results

**Gold benchmark (test macro F1):**

| Language  | Baseline | XLM-R     | MuRIL     | MuRIL+DAPT |
|-----------|----------|-----------|-----------|------------|
| English   | 0.707    | **0.768** | —         | —          |
| Tamil     | 0.630    | 0.614     | 0.635     | **0.642**  |
| Malayalam | 0.763    | 0.777     | **0.818** | —          |

**Gold-vs-silver cross-evaluation (XLM-R, macro F1; rows = training labels,
columns = evaluation labels):**

| Language  | gold→gold | gold→silver | silver→gold | silver→silver |
|-----------|-----------|-------------|-------------|---------------|
| English   | 0.768     | 0.568       | 0.589       | **0.874**     |
| Tamil     | 0.614     | **0.752**   | 0.583       | **0.841**     |
| Malayalam | 0.777     | 0.682       | 0.714       | **0.865**     |

Key findings:

1. **Transformers win where pretraining matches the text** (XLM-R on English;
   MuRIL on code-mixed Dravidian text); DAPT on scraped in-domain data adds a
   further Tamil gain. Hand-crafted rule overrides are inert (≤13 predictions
   changed per split).
2. **Tamil's gold labels are the bottleneck, not the models**: LLM–gold
   agreement is near chance (κ = 0.09), audited gold errors are unambiguous,
   and the *gold-trained* Tamil XLM-R agrees more with the silver labels
   (0.752) than with its own gold test set (0.614).
3. **LLM labels are a far more consistent target**: identical architectures
   reach 0.84–0.87 macro F1 on silver versus 0.61–0.78 on gold, with 90%
   human-verified label accuracy.
4. Cross-standard transfer gaps mirror per-language gold quality (smallest
   for Malayalam, largest for Tamil).

Full tables: [`metrics/model_comparison.md`](metrics/model_comparison.md),
[`metrics/cross_evaluation_report.md`](metrics/cross_evaluation_report.md),
audit record: [`metrics/human_audit.json`](metrics/human_audit.json).

---

## Project Structure

```
DSCI-601_2/
├── initial_data/          # Raw HopeEDI CSVs (semicolon format)
├── preprocess/
│   └── preprocess_all.py  # parse, clean, binarize, dedupe
├── training/
│   ├── train_all.py               # gold TF-IDF baseline (+ rules variants)
│   └── train_silver_baseline.py   # silver baseline + 2x2 cross-eval
├── evaluation/
│   ├── evaluate.py                # unified model evaluation
│   ├── predict.py                 # live 0/1 prediction demo
│   ├── compare_models.py          # gold-benchmark model table
│   └── cross_eval_report.py       # final gold-vs-silver matrices
├── data_collection/
│   ├── fetch_comments.py          # YouTube Data API collector (--lang)
│   ├── filter_corpus.py           # language filtering + cleaning
│   ├── llm_label_validate.py      # LLM-vs-gold agreement validation
│   ├── llm_label_corpus.py        # resumable batch labeling
│   └── build_silver_datasets.py   # contamination-controlled splits
├── notebooks/             # Colab: xlmr_finetune, muril_finetune,
│                          #        muril_dapt_tamil, xlmr_silver_crosseval
├── tests/                 # pytest suite (92 tests)
├── metrics/               # all metrics JSONs, report tables, human audit
├── docs/                  # Sphinx documentation
├── archive/               # previous-semester scripts (reference)
├── config.json            # paths, label maps, hyperparameters, rule patterns
├── requirements.txt
└── README.md
```

Generated data (`processed/`, `processed_silver/`, `models/`,
`data_collection/collected/`) is gitignored and reproducible. Raw scraped
comments are never committed. API keys live in a local `.env` (gitignored).

---

## Installation

```bash
git clone https://github.com/anjanakopparthi/DSCI-601_2.git
cd DSCI-601_2
pip install -r requirements.txt
```

Key dependencies: `pandas`, `scikit-learn`, `joblib`, `requests`, `pytest`,
`sphinx` (see `requirements.txt`). Transformer training additionally uses
`transformers`, `datasets`, `accelerate` (installed inside the Colab
notebooks).

---

## Dataset

**Source:** [HopeEDI](https://github.com/bharathichezhiyan/HopeEDI)
(Chakravarthi, 2020) — YouTube comments labeled `Hope_speech`,
`Non_hope_speech`, or `not-<language>`. Raw files are included in
`initial_data/`.

**Raw format** (semicolon-delimited single column):
```
text content;Hope_speech;
another text;Non_hope_speech;
```

**Processed format** (binary; not-in-language rows deleted):
```csv
text,label_str,label
"i have hope for tomorrow",Hope_speech,1
"this situation is terrible",Non_hope_speech,0
```

**Split sizes after preprocessing (gold):**

| Language  | Train  | Dev   | Test  | Hope% |
|-----------|--------|-------|-------|-------|
| English   | 21,810 | 2,818 | 2,823 |  8.7% |
| Tamil     | 13,954 | 1,751 | 1,752 | 44.3% |
| Malayalam |  7,661 |   970 |   965 | 19.5% |

**Silver datasets** (LLM-labeled scraped comments, built by Phase 3):

| Language  | Train  | Dev   | Test  | Hope% |
|-----------|--------|-------|-------|-------|
| English   | 20,949 | 2,619 | 2,619 | 66.7% |
| Tamil     |  9,855 | 1,232 | 1,232 | 64.7% |
| Malayalam |  6,627 |   828 |   829 | 59.8% |

---

## Quick Start — Gold Track

All scripts resolve paths relative to the repository root and run from any
working directory:

```bash
python preprocess/preprocess_all.py     # raw CSVs -> processed/
python training/train_all.py            # baselines -> models/, metrics/
python evaluation/evaluate.py           # evaluate saved models (--lang, --split)
python evaluation/predict.py tamil      # live demo (--interactive)
python evaluation/compare_models.py     # benchmark comparison table
```

Runtime: preprocessing ~1 min, baseline training ~3 min, evaluation seconds.

---

## Silver Track — Data Collection & LLM Labeling

Requires `YOUTUBE_API_KEY` and `GEMINI_API_KEY` in a repo-root `.env`
(both free tier).

```bash
# 1. Collect comments (official Data API; quota-aware, resumable)
python data_collection/fetch_comments.py --lang english   # + tamil, malayalam

# 2. Filter to the target language (script + romanized lexicons)
python data_collection/filter_corpus.py --lang english

# 3. Validate the LLM labeler against gold BEFORE trusting it
python data_collection/llm_label_validate.py --lang english --few-shot 30

# 4. Batch-label the corpus (resumable across quota days)
python data_collection/llm_label_corpus.py --lang english

# 5. Build contamination-controlled silver splits
python data_collection/build_silver_datasets.py

# 6. Train + cross-evaluate the silver baseline (2x2 matrix)
python training/train_silver_baseline.py

# 7. Aggregate all matrices into the final report
python evaluation/cross_eval_report.py
```

---

## Transformer Fine-tuning (Colab)

The notebooks in `notebooks/` run on a free Colab T4 GPU. They read
`processed/` and `processed_silver/` from Google Drive at
`MyDrive/hope_speech/` and write models + metrics JSONs back to Drive
(copy the metrics into `metrics/` afterwards):

| Notebook | Purpose | ~Time/lang |
|---|---|---|
| `xlmr_finetune.ipynb` | XLM-R on gold data | 15–45 min |
| `muril_finetune.ipynb` | MuRIL on gold data (Tamil/Malayalam) | 25–40 min |
| `muril_dapt_tamil.ipynb` | MLM domain-adaptive pretraining + fine-tune | 60 min |
| `xlmr_silver_crosseval.ipynb` | XLM-R on silver + built-in 2×2 cross-eval | 20–50 min |

All use class-weighted loss, fp16, early stopping, and best-epoch selection
by dev macro F1.

---

## Testing

```bash
python -m pytest tests/ -v      # expect: 92 passed
```

Covers the processed-data contract (strictly binary labels, no
not-in-language rows, no URL/mention artifacts, no duplicates, casing
preserved), training utilities (pipeline construction, rule overrides across
scripts and emoji, metrics), and saved-model integrity (including a
regression test for the historical rules-pickle corruption bug).

---

## Documentation

```bash
pip install sphinx sphinx-rtd-theme
cd docs && ./make.bat html          # or: make html
start build/html/index.html         # Windows
```

Sphinx autodoc generates the module reference from docstrings across
`preprocess/`, `training/`, `evaluation/`, and `data_collection/`.

---

## Ethics & Data Notes

- Comments collected via the **official YouTube Data API** within quota — no
  HTML scraping.
- **Anonymized at collection**: author names/ids are never read or stored;
  only comment text, like count, and video id (for dedup) are kept, and raw
  collected data stays out of version control.
- **Silver labels are LLM-generated and disclosed as such**; quality is
  quantified by per-language gold-agreement validation (0.545–0.855) and a
  manual audit (90/100 on a random English sample).
- Gold-label criticism is supported by concrete audited examples, not
  aggregate numbers alone.

---

## License

MIT License.

## Authors

**Anjana Kopparthi** — DSCI-601, Rochester Institute of Technology.

## Citation

```bibtex
@misc{kopparthi2026hope,
  title={Multilingual Hope Speech Detection: Benchmark Label Quality and an
         LLM-Labeled Silver Standard},
  author={Kopparthi, Anjana},
  year={2026}
}
```

Dataset: Chakravarthi, B. R. (2020). *HopeEDI: A multilingual hope speech
detection dataset for equality, diversity, and inclusion.* PEOPLES @ COLING.
