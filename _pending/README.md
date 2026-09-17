# Pending code (not used by the application)

Code drafted before the phased plan was adopted, kept here so it is not lost or rewritten. **Nothing in this folder
is imported by `app.py` or covered by the test suite.** Each file is moved into `src/` only during its phase, after
inspection, adaptation to the current interfaces and tests.

| File | Target phase | Origin | Verification state |
|---|---|---|---|
| src/quality/quality_engine.py, validators.py | 2 | research notebooks | tested there; not yet in this repo |
| src/utils/audit.py | 2 | research notebooks | tested there |
| src/preprocessing/cleaner.py, duplicates.py, missing_values.py, categorical.py, numerical.py, pipeline.py, outliers.py | 2–3 | research notebooks | tested there |
| src/preprocessing/split.py, sampling.py | 3 | research notebooks | tested there |
| src/preprocessing/levels.py | 3 | drafted for the E0/E1/E2 brief | **unverified draft** |
| src/features/behavioral_features.py | 4 | research notebooks | tested there (brute-force + point-in-time) |
| src/quality/leakage_detector.py | 4 | research notebooks | tested there |
| src/representation/transaction_formatter.py | 5/7 | research notebooks | tested there |
| src/representation/tabular_tokenizer.py, text_builder.py | 5/7 | drafted | **unverified draft** |
| src/models/sanity_transformer.py | 5 | drafted | **unverified draft** (binary only) |
| src/evaluation/metrics.py | 6 | research notebooks | tested there |
| src/models/hf_backend.py | 7 | research notebooks (Nemotron LoRA on T4) | tested there |
| src/models/hf_adapter.py, api_adapter.py, registry.py, base_draft.py | 7 | drafted | **unverified draft** |
| tests/synthetic.py | 3+ | drafted | copy now lives in src/ingestion/demo_data.py |
