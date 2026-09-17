# Transaction Data Intelligence

A research prototype for **automated intake, profiling, quality analysis, preprocessing and model readiness of
tabular data**, with a controlled comparison of how data preparation affects transformer training.

> **Research question.** How does systematic preprocessing and quality control affect transformer training
> compared with training on raw data? The framework is built to *measure* this. It does not assume that
> preprocessing helps.

**Status: Phase 1 of 8.** See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for exactly what works today.

## Why

Model results depend heavily on how data is prepared, yet preparation is often ad hoc and undocumented. This
project makes each preparation level explicit and reproducible:

| Level | Meaning | Status |
|---|---|---|
| E0 Raw | Minimal formatting only | Phase 3 |
| E1 Quality processed | Type, categorical, numeric and datetime processing fitted on training rows | Phase 3 |
| E2 Feature engineered | E1 + time features + leakage-safe history features | Phase 4 |

The same model, split, seed and settings are then trained on each level so the preparation level is the only variable.

## Architecture

Generic framework (reusable for any tabular dataset):

```text
Dataset → Ingestion → Schema detection → Profiling → Quality → Basic preprocessing
        → E0 / E1 / E2 → Generic tabular representation → Model → Task-specific evaluation
```

First case study: financial transactions → fraud classification → E0 vs E1 vs E2 → small transformer.
Column names are **detected or configured, never assumed**; tests rename every column to prove it.

```text
app.py                  Streamlit interface
config.yaml             all settings (column hints optional)
src/ingestion/          loading, metadata, schema detection, roles, task detection, synthetic demo data
src/profiling/          profiling
src/models/             model interface and task definitions (implementations in later phases)
src/utils/              configuration, hardware detection
src/quality, preprocessing, features, representation, evaluation/   filled in later phases
tests/                  pytest suite (includes a headless UI test)
_pending/               code drafted for later phases; NOT used by the app until verified in its phase
```

## Installation and local use (CPU is enough)

```bash
git clone <your-repository-url>
cd transaction-data-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the address Streamlit prints (usually http://localhost:8501) and use **Data upload**:

* upload a CSV / JSON / JSON Lines file (stored unchanged in `data/raw/`), or
* load a large file from a local path (avoids browser upload limits), or
* load the **synthetic** demo file (generated; not real data).

Run the tests:

```bash
pytest
```

A GPU is **not** required. PyTorch is used to detect CUDA; on a CPU-only machine you may install the smaller CPU
build: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Supported data

Structured tabular data as CSV, JSON (array of records) or JSON Lines. Detected column roles:
`TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT, UNKNOWN`.

Tasks: binary classification, multiclass classification, regression (detected from the target and overridable).
The first complete implementation focuses on binary classification.

## Schema detection

Roles are inferred from **values first and names second**: datetime parse rates, uniqueness, repetition per
entity, card-number checksums, epoch-time ranges, category cardinality, and target-like characteristics
(binary 0/1, imbalance, name hints). Every detection can be overridden in the Profiling section or in `config.yaml`.

## Profiling

Row, missing-value and duplicate counts; numeric statistics, skewness and outlier counts (reported, never removed);
categorical cardinality, rare categories and case/whitespace variants; target distribution; time range and monthly
volume; entity statistics; early leakage indicators. Values of columns detected as personal data are masked.

## Current case study

Credit-card transactions with a binary `is_fraud` target (about 0.58% positive), a card identifier as entity and a
transaction timestamp. The framework detects these roles from the data; they are not hard-coded.

## Leakage prevention, experiments, models and Colab

Documented as each phase is implemented: temporal splitting, train-only fitting, point-in-time history features,
the built-in transformer (a **small transformer trained from scratch for tabular supervised learning, not a
pretrained language model**), optional Hugging Face / Nemotron / API backends, and `notebooks/colab_demo.ipynb`
for optional GPU acceleration.

## Limitations (current)

* Only Phase 1 is implemented: no preprocessing, training or evaluation yet.
* Very large files are loaded fully into memory; profiling offers a sample above 500,000 rows.
* Schema detection is heuristic; always review detected roles.

## Future extensions

Additional datasets (churn, credit risk), multiclass and regression end to end, database connectors, streaming,
experiment tracking, drift detection, privacy scanning, deployment.
