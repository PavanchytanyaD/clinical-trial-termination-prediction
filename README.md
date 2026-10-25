# Predicting Early Termination of Clinical Trials

MLOps course project (Northeastern University). The system predicts the probability that a registered clinical trial will be stopped early (terminated, withdrawn, or suspended) from its design, and an agentic AI system explains why a flagged trial is at risk by citing similar past trials that were stopped.

## Approach

- **Data:** ClinicalTrials.gov, accessed through the AACT database (monthly PostgreSQL snapshot). One row is one trial, identified by its NCT ID.
- **Baseline:** a classical ML model (gradient-boosted trees or logistic regression) on the structured design features.
- **Main model:** BioBERT text embeddings (not fine-tuned) combined with the structured features in a neural network with a classification head.
- **Agentic system:** looks up similar past stopped trials in the structured data and a text index, reads their recorded stop reasons, and writes a cited explanation.
- **Platform:** Google Cloud (Vertex AI Pipelines, BigQuery, Cloud Storage, Artifact Registry, GKE, Vertex AI Vector Search), with GitHub Actions, DVC, and MLflow.

## Data pipeline (assignment 2)

The data pipeline runs every month: it fetches an AACT snapshot, builds the features and the BioBERT text embeddings, checks the data, reports bias, and saves a versioned copy of everything. Airflow orchestrates it and DVC versions it.

```
               +--> preprocess --+--> schema_stats --> validate --+
               |                 +--> anomalies -------------------+
    acquire ---+--> bias ------------------------------------------+--> publish
               +--> text --> embeddings (Vertex AI GPU) -----------+
```

It runs locally (`dvc repro`, or Airflow in Docker) and on Google Cloud (the same Airflow on a Compute Engine VM, or Cloud Composer), with the data versions in a GCS bucket and the embeddings computed on a Vertex AI T4 GPU. See [Data-Pipeline/README.md](Data-Pipeline/README.md#where-it-runs).

| Guideline item | Where | Owner |
|---|---|---|
| 1. Data acquisition | `src/ingestion/fetch_aact.py` | Nithish |
| 2. Data preprocessing | `src/features/` | Nithish |
| 3. Test modules | `tests/`, `Data-Pipeline/tests/` | everyone |
| 4. Pipeline orchestration (Airflow) | `Data-Pipeline/dags/ctt_data_pipeline.py` | Pavanchytanya |
| 5. Data versioning (DVC) | `Data-Pipeline/dvc.yaml`, `dvc.lock`, `.dvc/` | Pavanchytanya |
| 6. Tracking and logging | `Data-Pipeline/scripts/tracking.py`, `alerts.py`, `src/logging_utils.py` | Sivapriya (with Pavanchytanya) |
| 7. Schema and statistics | `Data-Pipeline/scripts/schema_stats.py`, `validate.py` | Sivapriya (with Pavanchytanya) |
| 8. Anomaly detection and alerts | `Data-Pipeline/scripts/anomalies.py` | Pavanchytanya |
| 9. Pipeline flow optimization | `Data-Pipeline/docs/pipeline_optimization.md` | Pavanchytanya |
| Text and BioBERT embeddings stages | `src/features/build_text_sample.py`, `biobert_embeddings.py` (Nithish); `Data-Pipeline/scripts/text.py`, `embeddings.py`, Vertex AI job (Pavanchytanya) | Nithish, Pavanchytanya |
| Cloud hosting | `Data-Pipeline/airflow/`, `gce/`, `composer/`, `docker/` | Pavanchytanya |
| Bias detection and slicing | `src/bias/slicing.py`, `slice_report.py`, `model_slices.py` | Danush, Atharva |
| Bias mitigation and report | `src/bias/mitigate.py`, `docs/bias_report.md` | Atharva, Danush |
| Error handling, code style, contributing | `src/error_handling.py`, `.flake8`, `CONTRIBUTING.md` | Sathyasai |

Every stage is explained in detail in [Data-Pipeline/README.md](Data-Pipeline/README.md); deployment to Cloud Composer is in [Data-Pipeline/composer/README.md](Data-Pipeline/composer/README.md).

## Code structure

```
clinical-trial-termination-prediction/
|- Data-Pipeline/            the data pipeline (folder layout from the assignment guidelines)
|  |- dags/                  Airflow DAG
|  |- scripts/               one module per stage, plus tracking, alerts and the DVC stage runner
|  |- tests/                 tests for the pipeline stages, DAG and DVC runner
|  |- data/                  raw snapshot and processed features (DVC-tracked, not in git)
|  |- reports/               statistics, validation, anomaly and bias reports (DVC-tracked)
|  |- schema/baseline/       frozen schema and statistics new snapshots are validated against
|  |- logs/                  step logs and run records (not in git)
|  |- docs/                  pipeline optimization note and Gantt charts
|  |- docker/, composer/     container image and Cloud Composer deployment
|  |- dvc.yaml, params.yaml  DVC stages and their parameters
|  \- README.md
|- src/                      shared code the pipeline stages call
|  |- ingestion/             AACT download / database export and table checks
|  |- features/              text cleaning, structured features, BioBERT embeddings
|  |- bias/                  data slicing, bias reports, mitigation
|  |- error_handling.py      typed pipeline errors and stage logging
|  \- logging_utils.py       console and file logging for src scripts
|- configs/bias_config.yaml  bias slicing and mitigation settings
|- tests/                    tests for src/
|- docs/                     scoping document, diagrams, bias report
|- requirements*.txt         dependencies (see Setup)
\- CONTRIBUTING.md           logging, error handling, code style, branches
```

## Setup

Prerequisites: Python 3.11 or later (numpy 2.3 and pandas 3 do not support 3.10), Git, and for the cloud run Docker and the Google Cloud CLI.

```bash
git clone https://github.com/PavanchytanyaD/clinical-trial-termination-prediction.git
cd clinical-trial-termination-prediction
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r Data-Pipeline/requirements.txt   # the data pipeline (includes DVC, Great Expectations, bias)
pip install -r requirements.txt                 # src/ including PyTorch and BioBERT (optional for the pipeline)
pip install -r requirements-dev.txt             # flake8, black, pytest
cp Data-Pipeline/.env.example Data-Pipeline/.env   # optional: SMTP settings for email alerts, AACT account
```

## Run the pipeline

Without an AACT account the pipeline runs on a seeded placeholder snapshot in the AACT format, so every step can be run and tested anywhere.

```bash
cd Data-Pipeline
dvc repro            # runs every stage that changed; a second run skips them all
dvc metrics show     # validation and anomaly summaries
```

- Real data: set `acquire.source` in `Data-Pipeline/params.yaml` to `zip` (with `acquire.zip`, the AACT monthly export) or `db` (with `AACT_USER` and `AACT_PASSWORD` set).
- With Airflow on a laptop: `CTT_RUNNER=local airflow dags test ctt_data_pipeline <date>` (see [Data-Pipeline/README.md](Data-Pipeline/README.md)).
- On Google Cloud: Cloud Composer runs the same DAG, one container per task ([Data-Pipeline/composer/README.md](Data-Pipeline/composer/README.md)).

## Data versioning and reproducibility

- `Data-Pipeline/dvc.yaml` declares each stage's code, inputs, parameters and outputs. `dvc.lock` stores the md5 of every output, so each commit of `dvc.lock` is one exact data version; the data itself goes to the DVC remote (a GCS bucket in the `ctt-mlops` project) with `dvc push`.
- Return to any data version: `git checkout <commit> -- Data-Pipeline/dvc.lock && cd Data-Pipeline && dvc pull && dvc checkout`.
- The placeholder snapshot is seeded, so a fresh clone gives the same data and the same `dvc.lock` hashes for the stage outputs.
- Fresh-clone check: clone, install `Data-Pipeline/requirements.txt`, run `pytest` in the repository root and in `Data-Pipeline/`, then `dvc repro` in `Data-Pipeline/`.

## Tests

```bash
pytest                          # src/ code
cd Data-Pipeline && pytest      # pipeline stages, DAG and DVC runner (DAG tests need Airflow)
```

## Component details (data acquisition and preprocessing)

Each module can also be run on its own from the repository root. It logs to the console and to `logs/<step>.log`, and exits with code 1 if it fails.

### 1. Data acquisition

`src/ingestion/fetch_aact.py` saves one AACT snapshot to `data/raw/aact/<YYYYMMDD>/`, keeping only the 12 tables the project uses, and writes a `manifest.json` (source, snapshot date, file sizes) so the snapshot can be versioned with DVC. Both sources need a free [AACT account](https://aact.ctti-clinicaltrials.org/).

- From the monthly flat-file export (a downloaded zip, or its URL):
  ```bash
  python -m src.ingestion.fetch_aact --zip 20261001_export_ctgov.zip
  ```
- From the AACT cloud database, with the account in environment variables:
  ```bash
  export AACT_USER=<username>
  export AACT_PASSWORD=<password>
  python -m src.ingestion.fetch_aact --db
  ```

Each table is checked for an `nct_id` column and at least one row. Downloads retry up to 3 times and never leave a partial file.

### 2. Text preprocessing

`src/features/text_cleaning.py` cleans the free-text fields: it turns AACT's `~` back into line breaks, removes markdown escapes (`\>=` becomes `>=`), normalizes Unicode (ligatures, full-width characters), collapses whitespace and drops boilerplate lines. It also splits `eligibilities.criteria` into inclusion and exclusion items, handling header variants such as "Key Inclusion Criteria" and "Non-inclusion criteria".

`src/features/build_text_sample.py` builds one row of cleaned text per trial: title, brief summary, detailed description, inclusion criteria, exclusion criteria and planned outcomes (from `design_outcomes`, which are written at registration, not results).

```bash
# All trials
python -m src.features.build_text_sample --export-dir data/raw/aact/20261001 --all --out data/processed/text_all.jsonl
# 10k sample of finished trials, for the BioBERT test run
python -m src.features.build_text_sample --export-dir data/raw/aact/20261001 --n 10000
```

### 3. Structured preprocessing and features

`src/features/structured_features.py` cleans the structured fields and builds 155 features per trial: phase and study type, design (allocation, masking, purpose), arms, age range, gender, inclusion and exclusion counts, lead sponsor class and collaborators, sites and countries, intervention types, conditions and MeSH therapeutic areas.

```bash
python -m src.features.structured_features --export-dir data/raw/aact/20261001
```

- **Label:** from `overall_status`: 1 for terminated, withdrawn or suspended, 0 for completed, and empty for trials still running (left out of training).
- **Encoding:** categoricals are one-hot encoded against fixed category lists, so every snapshot gets the same columns, with `_unlisted` (a value outside the list) and `_missing` columns. The catch-all is not called `_other` because several AACT fields have a real `OTHER` value. Numeric fields get a `_missing` flag and are filled with 0, so nothing is fitted on the full data.
- **Leakage:** fields filled in after a trial ends are excluded: actual enrollment, completion dates, start date type, `why_stopped` and results. `n_facilities`, `has_us_facility`, `has_single_facility`, `n_countries` and the `_missing` flags of the facility columns are kept but flagged as a leakage risk, because sites are often removed from withdrawn trials. Both lists are in the output's `.meta.json`.

Output: `data/processed/structured_features.csv`, with `nct_id`, `overall_status`, `label`, `start_date`, `study_first_submitted_date` and the features, and `structured_features.meta.json` with the feature list and label counts.

### 4. BioBERT embeddings

`src/features/biobert_embeddings.py` turns text into 768-number vectors with BioBERT (`dmis-lab/biobert-base-cased-v1.2`, not fine-tuned). Text longer than BioBERT's 512-token limit is split into overlapping 512-token windows (64 tokens of overlap). Each window is mean-pooled, and the windows are averaged, weighted by length, so no text is cut off. Empty text gets a zero vector. If the GPU runs out of memory, the batch is split in half and retried.

`src/features/biobert_test_run.py` runs this on the 10k sample and reports token lengths and run time per field:

```bash
python -m src.features.biobert_test_run --sample data/processed/text_sample_10k.jsonl
```

On an RTX 4060 laptop GPU the 10k sample takes about 2.7 minutes, so all finished trials take about 2 hours. In the pipeline, the `embeddings` stage runs the same code on every trial as a Vertex AI job on a T4 GPU (`Data-Pipeline/scripts/embeddings.py`), or locally with `embeddings.backend: local`. On Windows, `requirements.txt` installs the CPU build of PyTorch; install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) to use a GPU.


### 5. Bias detection and mitigation

`src/bias/` slices the finished trials by phase, study type, sponsor class, region, therapeutic area, gender, age group and start era, flags groups that are too small or whose stopped rate is far from the overall rate, and reweights or resamples under-represented groups. In the pipeline it runs as the `bias` stage on the versioned snapshot; on its own it can also query the AACT database:

```bash
python -m src.bias.run_bias_detection --config configs/bias_config.yaml
python -m src.bias.run_bias_mitigation --config configs/bias_config.yaml
```

Findings, the mitigation and its trade-offs are in [docs/bias_report.md](docs/bias_report.md). The analysis is at the data level; model-level evaluation per slice comes with the model development assignment.

## Team

- Atharva Nilesh Mahadik
- Danush Gopinath
- Nithish Bhat
- Pavanchytanya Dharmapuri
- Sathyasai Rajan
- Sivapriya Sugumaran

## Data rights

ClinicalTrials.gov and AACT (CTTI) data are public. The data is trial-level and contains no patient-level personal data.
