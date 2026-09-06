Multilingual Hope Speech Detection
==================================

Binary hope speech classification (``0 = Non_hope_speech``, ``1 = Hope_speech``)
for **English**, **Tamil**, and **Malayalam** YouTube comments, based on the
HopeEDI dataset (LT-EDI shared task) — extended with an investigation of the
benchmark's label quality and an LLM-labeled **silver-standard** re-annotation.

The project has three phases:

1. **Gold pipeline.** A rebuilt binary preprocessing/training/evaluation
   pipeline and a TF-IDF + Logistic Regression baseline, followed by
   fine-tuned **XLM-R**, **MuRIL**, and **MuRIL + domain-adaptive pretraining
   (DAPT)** on 12k scraped in-domain Tamil comments.
2. **Label-quality investigation.** LLM labeling was validated against gold
   before use. Agreement tracked known annotation quality per subset —
   Malayalam 0.855, English 0.782 (few-shot), Tamil 0.545 (κ = 0.09) — and a
   manual audit of Tamil disagreements found gold ``Hope_speech`` labels on
   insults, jokes, and neutral questions, explaining the ~0.63 macro-F1
   ceiling all gold-trained models converge to.
3. **Silver track.** 46,875 topic-matched YouTube comments were collected via
   the official Data API, filtered per language (script detection + romanized
   lexicons), labeled with the validated LLM setup
   (``gemini-3.1-flash-lite`` + 30 gold few-shot examples), audited by hand
   (**90/100 human-verified accuracy** on a random English sample), split with
   gold-contamination control, and used to retrain both model families for a
   full gold-vs-silver cross-evaluation.

Results — gold benchmark (test macro F1)
----------------------------------------

.. list-table::
   :header-rows: 1

   * - Language
     - Baseline
     - XLM-R
     - MuRIL
     - MuRIL + DAPT
   * - English
     - 0.707
     - **0.768**
     - —
     - —
   * - Tamil
     - 0.630
     - 0.614
     - 0.635
     - **0.642**
   * - Malayalam
     - 0.763
     - 0.777
     - **0.818**
     - —

Results — gold vs silver cross-evaluation (XLM-R, macro F1)
-----------------------------------------------------------

Rows = training labels, columns = evaluation labels.

.. list-table::
   :header-rows: 1

   * - Language
     - gold → gold
     - gold → silver
     - silver → gold
     - silver → silver
   * - English
     - 0.768
     - 0.568
     - 0.589
     - **0.874**
   * - Tamil
     - 0.614
     - **0.752**
     - 0.583
     - **0.841**
   * - Malayalam
     - 0.777
     - 0.682
     - 0.714
     - **0.865**

Key findings
------------

- **Transformers beat the baseline where their pretraining matches the text**
  (XLM-R on English; MuRIL on the code-mixed Dravidian languages), and DAPT on
  scraped in-domain text gives a further gain on Tamil.
- **The Tamil gold labels are the bottleneck, not the models.** LLM-gold
  agreement is barely above chance (κ = 0.09) with audited examples of
  mislabeled gold; strikingly, the *gold-trained* Tamil XLM-R agrees more
  with the silver labels (0.752) than with its own gold test set (0.614).
- **LLM labels form a substantially more consistent target**: the same
  architectures reach 0.84–0.87 macro F1 on silver data versus 0.61–0.78 on
  gold, with 90 % human-verified label accuracy on audit.
- Cross-standard transfer gaps mirror per-language gold quality
  (smallest for Malayalam, largest for Tamil).

Pipeline
--------

Gold track (local)::

    python preprocess/preprocess_all.py        # raw CSVs -> processed/ (binary)
    python training/train_all.py               # TF-IDF baselines -> models/
    python evaluation/evaluate.py              # evaluate saved models
    python evaluation/predict.py tamil         # live 0/1 demo
    python evaluation/compare_models.py        # gold-benchmark model table
    python -m pytest tests/                    # test suite

Silver track (local + Colab)::

    python data_collection/fetch_comments.py --lang english     # collect
    python data_collection/filter_corpus.py  --lang english     # clean/filter
    python data_collection/llm_label_validate.py --lang english --few-shot 30
    python data_collection/llm_label_corpus.py --lang english   # batch label
    python data_collection/build_silver_datasets.py             # silver splits
    python training/train_silver_baseline.py                    # TF-IDF 2x2
    python evaluation/cross_eval_report.py                      # final matrices

Transformer fine-tuning runs on Google Colab (T4) via ``notebooks/``:
``xlmr_finetune.ipynb``, ``muril_finetune.ipynb``, ``muril_dapt_tamil.ipynb``,
``xlmr_silver_crosseval.ipynb``.

Module reference
----------------

Preprocessing (``preprocess/preprocess_all.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: preprocess_all
   :members:
   :undoc-members:

Baseline training (``training/train_all.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: train_all
   :members:
   :undoc-members:

Silver baseline + cross-eval (``training/train_silver_baseline.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: train_silver_baseline
   :members:
   :undoc-members:

Evaluation (``evaluation/evaluate.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evaluate
   :members:
   :undoc-members:

Prediction demo (``evaluation/predict.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: predict
   :members:
   :undoc-members:

Model comparison (``evaluation/compare_models.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: compare_models
   :members:
   :undoc-members:

Cross-evaluation report (``evaluation/cross_eval_report.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cross_eval_report
   :members:
   :undoc-members:

Comment collection (``data_collection/fetch_comments.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fetch_comments
   :members:
   :undoc-members:

Corpus filtering (``data_collection/filter_corpus.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: filter_corpus
   :members:
   :undoc-members:

LLM label validation (``data_collection/llm_label_validate.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: llm_label_validate
   :members:
   :undoc-members:

Batch LLM labeling (``data_collection/llm_label_corpus.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: llm_label_corpus
   :members:
   :undoc-members:

Silver dataset construction (``data_collection/build_silver_datasets.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: build_silver_datasets
   :members:
   :undoc-members:

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
