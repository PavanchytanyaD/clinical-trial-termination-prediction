# Predicting Early Termination of Clinical Trials

MLOps course project (Northeastern University). The system predicts the probability that a registered clinical trial will be stopped early (terminated, withdrawn, or suspended) from its design, and an agentic AI system explains why a flagged trial is at risk by citing similar past trials that were stopped.

## Approach

- **Data:** ClinicalTrials.gov, accessed through the AACT database (monthly PostgreSQL snapshot). One row is one trial, identified by its NCT ID.
- **Baseline:** a classical ML model (gradient-boosted trees or logistic regression) on the structured design features.
- **Main model:** BioBERT text embeddings (not fine-tuned) combined with the structured features in a neural network with a classification head.
- **Agentic system:** looks up similar past stopped trials in the structured data and a text index, reads their recorded stop reasons, and writes a cited explanation.
- **Platform:** Google Cloud (Vertex AI Pipelines, BigQuery, Cloud Storage, Artifact Registry, GKE, Vertex AI Vector Search), with GitHub Actions, DVC, and MLflow.

## Repository structure

```
clinical-trial-termination-prediction/
|- data/        raw and processed data (DVC-tracked, not committed)
|- src/         ingestion, features, model, and agent code
|- pipelines/   pipeline definitions (ingest, train, evaluate, deploy)
|- tests/       unit and integration tests
|- docs/        project scoping document and diagrams
|- logs/        pipeline logs (not committed)
|- README.md
```

## Architecture

Training pipeline:

![Training pipeline](docs/diagrams/2_gcp_training.png)

Inference pipeline:

![Inference pipeline](docs/diagrams/3_gcp_inference.png)

## Installation (tentative)

These steps are a plan and will be finalized as the code is built.

**Prerequisites:** Python 3.11 or later (numpy 2.3 and pandas 3 do not support 3.10), Git, Docker, DVC, and the Google Cloud CLI with access to the project's GCP account.

1. Clone the repository:
   ```bash
   git clone https://github.com/PavanchytanyaD/clinical-trial-termination-prediction.git
   cd clinical-trial-termination-prediction
   ```
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Sign in to Google Cloud:
   ```bash
   gcloud auth application-default login
   ```
4. Pull the versioned data:
   ```bash
   dvc pull
   ```

## Usage (tentative)

These guidelines are a plan and will be updated with the exact commands as each pipeline is built.

- **Data pipeline:** downloads the monthly AACT snapshot, cleans it, builds the structured features and BioBERT embeddings, and writes them to BigQuery.
- **Training pipeline:** runs on Vertex AI Pipelines on a set schedule or when a new data version arrives. It trains the baseline and the neural network, evaluates them, and approves a model only if it passes the evaluation gate.
- **Inference:** a FastAPI service on GKE takes a trial (NCT ID), returns the probability that it stops early, and gives a short explanation that cites similar past trials.
- **Tests:** run with `pytest` from the repository root.

## Data pipeline

Each step is a Python module run from the repository root. Every step logs to the console and to `logs/<step>.log`, and exits with code 1 if it fails, so an orchestrator can detect the failure.

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

`src/features/structured_features.py` cleans the structured fields and builds 151 features per trial: phase and study type, design (allocation, masking, purpose), arms, age range, gender, inclusion and exclusion counts, lead sponsor class and collaborators, sites and countries, intervention types, conditions and MeSH therapeutic areas.

```bash
python -m src.features.structured_features --export-dir data/raw/aact/20261001
```

- **Label:** from `overall_status`: 1 for terminated, withdrawn or suspended, 0 for completed, and empty for trials still running (left out of training).
- **Encoding:** categoricals are one-hot encoded against fixed category lists, so every snapshot gets the same columns, with `_other` and `_missing` columns. Numeric fields get a `_missing` flag and are filled with 0, so nothing is fitted on the full data.
- **Leakage:** fields filled in after a trial ends are excluded: actual enrollment, completion dates, start date type, `why_stopped` and results. `n_facilities`, `has_us_facility`, `has_single_facility` and `n_countries` are kept but flagged as a leakage risk, because sites are often removed from withdrawn trials. Both lists are in the output's `.meta.json`.

Output: `data/processed/structured_features.csv`, with `nct_id`, `overall_status`, `label`, `start_date`, `study_first_submitted_date` and the features, and `structured_features.meta.json` with the feature list and label counts.

### 4. BioBERT embeddings

`src/features/biobert_embeddings.py` turns text into 768-number vectors with BioBERT (`dmis-lab/biobert-base-cased-v1.2`, not fine-tuned). Text longer than BioBERT's 512-token limit is split into overlapping 512-token windows (64 tokens of overlap). Each window is mean-pooled, and the windows are averaged, weighted by length, so no text is cut off. Empty text gets a zero vector. If the GPU runs out of memory, the batch is split in half and retried.

`src/features/biobert_test_run.py` runs this on the 10k sample and reports token lengths and run time per field:

```bash
python -m src.features.biobert_test_run --sample data/processed/text_sample_10k.jsonl
```

On an RTX 4060 laptop GPU the 10k sample takes about 2.7 minutes, so all finished trials take about 2 hours, and the full embedding is run locally on a GPU. On Windows, `requirements.txt` installs the CPU build of PyTorch; install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) to use a GPU.

### Tests

`pytest` from the repository root runs the unit tests for text cleaning, the eligibility split, the text table, the structured features, the embeddings (with a tiny local model, so no download is needed) and the AACT fetch.

## Team

- Atharva Nilesh Mahadik
- Danush Gopinath
- Nithish Bhat
- Pavanchytanya Dharmapuri
- Sathyasai Rajan
- Sivapriya Sugumaran

## Data rights

ClinicalTrials.gov and AACT (CTTI) data are public. The data is trial-level and contains no patient-level personal data.
