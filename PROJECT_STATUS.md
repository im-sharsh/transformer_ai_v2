# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 1 — Foundation: complete.** Waiting for instruction before starting Phase 2.

## Completed work (Phase 1)

- Repository structure, `requirements.txt`, `config.yaml`, `.gitignore`, `.streamlit/config.toml`, README
- Streamlit app (`app.py`) with sections Dashboard, Data Upload, Profiling, Quality Analysis, Processing, Model,
  Experiments, Results; pipeline tracker in the sidebar; clear error display
- CPU / CUDA detection with honest recommendations (`src/utils/hardware.py`)
- CSV / JSON / JSON Lines intake by upload, local path, or synthetic demo generator; raw files never modified;
  SHA-256 dataset IDs and metadata
- Generic schema detection and framework roles TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT,
  UNKNOWN; candidate detection for target, entity and datetime; task detection (binary / multiclass / regression);
  overrides in the UI and config (`src/ingestion/roles.py`)
- Profiling: overview, numeric statistics and skewness, categorical cardinality and rare categories, target
  distribution, time range and monthly volume, entity statistics, early leakage indicators; optional reproducible
  sample above 500,000 rows (counts always use all rows)
- Interfaces for later phases: `src/models/base.py` (ModelAdapter), `src/models/tasks.py` (task specs)
- Quality, Processing, Model, Experiments and Results sections show "Not yet implemented" / "Not run"

## Files created

app.py, config.yaml, requirements.txt, README.md, PROJECT_STATUS.md, .gitignore, .streamlit/config.toml,
src/ingestion/roles.py, src/ingestion/demo_data.py, src/utils/config.py, src/utils/hardware.py, src/models/base.py,
src/models/tasks.py, tests/conftest.py, tests/test_roles.py, tests/test_ingestion.py, tests/test_profiling.py,
tests/test_config_hardware.py, tests/test_app.py, _pending/README.md, package `__init__.py` files, .gitkeep files

## Files carried over (from the earlier research notebooks, used in Phase 1)

src/ingestion/loader.py, src/ingestion/metadata.py, src/ingestion/schema_detector.py, src/profiling/profiler.py

## Pending code (not used yet)

`_pending/` holds code for Phases 2–7 (research-notebook modules and unverified drafts). See `_pending/README.md`
for each file's target phase and verification state. **Do not rewrite these from scratch; inspect and move them.**

## Tests completed

`pytest` → **17 passed** (run in the build environment: Python 3.12, pandas 2.x, Streamlit 1.64, CPU only).

| File | Covers |
|---|---|
| test_ingestion.py | CSV / JSON / JSONL loading, raw file unchanged, metadata, bad input errors |
| test_roles.py | transaction roles, **renamed columns (no name dependence)**, churn dataset, task detection, overrides, summaries, profiling overrides |
| test_profiling.py | profile of transaction data and of a churn dataset, personal-data masking |
| test_config_hardware.py | config load/merge, hardware detection keys |
| test_app.py | headless UI: all sections render; demo data → schema → profile → dashboard |

Also verified: `streamlit run app.py` starts and the health endpoint returns ok.
**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in the build environment).

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Current git commit

`e1d1389` — Phase 1: foundation (this status file is added in the following commit)

## Known issues

- Large files are read fully into memory (the 1.3M-row file needs ~1 GB RAM plus the file size during upload).
- Browser upload is limited to 1 GB (`.streamlit/config.toml`); use "Load from path" for bigger files.
- Profiling the full 1.3M rows is expected to take on the order of a minute on a laptop (not yet measured).
- The IBM Plex Sans web font needs internet access; without it the UI falls back to system fonts.

## Next task (Phase 2 — quality analysis and basic preprocessing)

1. Read this file; run `pytest`.
2. Inspect `_pending/src/quality/*`, `_pending/src/utils/audit.py`, `_pending/src/preprocessing/{cleaner,duplicates,missing_values,categorical,numerical,outliers}.py`.
3. Move them into `src/`, adapt to the generic roles (`DatasetRoles`), add tests.
4. Build the Quality Analysis section and the preprocessing report (never fabricate values; report "No imputation required" when there are no missing values).

## TODO by phase

- [x] Phase 1: foundation
- [ ] Phase 2: quality analysis + basic preprocessing
- [ ] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [ ] Phase 4: feature engineering + leakage checks
- [ ] Phase 5: built-in CPU transformer
- [ ] Phase 6: evaluation + experiment runner
- [ ] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
