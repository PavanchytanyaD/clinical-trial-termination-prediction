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

The same code and DAG run in two places:

- **Google Cloud** (our deployment, project `ctt-mlops-project`): GitHub Actions tests every push and builds the container images into Artifact Registry. Cloud Composer orchestrates and runs each step as a pod from that image. Every data version is kept in a GCS bucket through DVC, and the embeddings run as a Vertex AI T4 GPU job. [docs/cloud_deployment.md](docs/cloud_deployment.md) shows our deployment with screenshots from a run and how to deploy it in your own project.
- **Your machine**: Airflow in Docker with the same DAG. The run state and the DVC remote are local folders and BioBERT runs on the CPU, so no cloud account is needed. See [Data-Pipeline/local/README.md](Data-Pipeline/local/README.md).

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
| Cloud hosting, local Docker Airflow and CI/CD | `.github/workflows/ci.yml`, `Data-Pipeline/composer/`, `Data-Pipeline/local/`, `docker/` | Pavanchytanya |
| Bias detection and slicing | `src/bias/slicing.py`, `slice_report.py`, `model_slices.py` | Danush, Atharva |
| Bias mitigation and report | `src/bias/mitigate.py`, `docs/bias_report.md` | Atharva, Danush |
| Error handling, code style, contributing | `src/error_handling.py`, `.flake8`, `CONTRIBUTING.md` | Sathyasai |

Every stage is explained in detail in [Data-Pipeline/README.md](Data-Pipeline/README.md); CI is in [.github/workflows/ci.yml](.github/workflows/ci.yml), Cloud Composer in [Data-Pipeline/composer/README.md](Data-Pipeline/composer/README.md) and the local Docker Airflow in [Data-Pipeline/local/README.md](Data-Pipeline/local/README.md).

## Code structure

```
clinical-trial-termination-prediction/
|- .github/workflows/ci.yml  CI/CD: tests on every push, then images -> Artifact Registry
|- Data-Pipeline/            the data pipeline (folder layout from the assignment guidelines)
|  |- dags/                  Airflow DAG
|  |- scripts/               one module per stage, plus tracking, alerts and the DVC stage runner
|  |- tests/                 tests for the pipeline stages, DAG and DVC runner
|  |- schema/baseline/       how the validation baseline (kept in GCS) is frozen
|  |- docs/                  pipeline optimization note and Gantt charts
|  |- composer/              Cloud Composer: setup, up, trigger, down; image builds
|  |- local/                 Airflow in Docker on one machine: up, trigger, down
|  |- docker/                pipeline image (Composer pods), BioBERT GPU image (Vertex AI)
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
|- docs/                     scoping document, cloud deployment (with screenshots), setup guide, bias report
|- requirements*.txt         dependencies of the images and the tests
\- CONTRIBUTING.md           logging, error handling, code style, branches
```

## Environment setup

| To | You need |
|---|---|
| Run the pipeline on your machine | Git, Bash and Docker with Compose v2 (Docker Desktop, or Docker Engine on Linux), with about 4 GB of memory for Docker. The image installs every Python dependency (`Data-Pipeline/requirements.txt`) and the BioBERT weights itself. |
| Run it on Google Cloud | Git, Bash, the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and a project with billing enabled. Cloud Build builds the images from `Data-Pipeline/docker/`. |
| Run the tests | Python 3.11 or later: `pip install -r requirements.txt -r requirements-bias.txt -r Data-Pipeline/requirements.txt -r requirements-dev.txt` |
| Use real AACT data or email alerts (optional) | `cp Data-Pipeline/.env.example Data-Pipeline/.env`, then fill it in with [docs/SETUP.md](docs/SETUP.md). Without it, the pipeline runs on a seeded test snapshot in the AACT format. |

Dependencies are pinned: `requirements.txt` (shared code), `requirements-bias.txt` (bias code), `Data-Pipeline/requirements.txt` (pipeline stages), `Data-Pipeline/docker/requirements-gpu.txt` (BioBERT GPU image) and `requirements-dev.txt` (tests and style check).

## Run the pipeline

Pick one of the two ways. Both run the same DAG on the seeded test snapshot
by default. With AACT credentials in `Data-Pipeline/.env`
([docs/SETUP.md](docs/SETUP.md)), both can run on the live AACT database too
(`SOURCE=db`).

**On your machine** (Docker; no cloud account):

```bash
git clone https://github.com/PavanchytanyaD/clinical-trial-termination-prediction.git
cd clinical-trial-termination-prediction/Data-Pipeline
./local/pipeline.sh up           # Airflow at http://localhost:8080 (admin / admin)
./local/pipeline.sh trigger      # one run; results in local/output/
./local/pipeline.sh down
```

Details: [Data-Pipeline/local/README.md](Data-Pipeline/local/README.md).

**On Google Cloud, in your own project** (gcloud CLI; billing enabled):

```bash
cd clinical-trial-termination-prediction/Data-Pipeline
gcloud auth login && export PROJECT=YOUR_PROJECT_ID
./composer/deploy.sh setup       # once: APIs, bucket, service accounts, roles
./composer/deploy.sh image && ./composer/deploy.sh gpu-image
./composer/deploy.sh up          # create Composer (about 25 min)
./composer/deploy.sh trigger     # one run
./composer/deploy.sh down        # delete Composer (billing stops)
```

Details, and our own deployment with screenshots:
[docs/cloud_deployment.md](docs/cloud_deployment.md). On our project
(`ctt-mlops-project`), CI builds the images on every push, so a session is
only `up`, `trigger` and `down`.

## Data versioning and reproducibility

- `Data-Pipeline/dvc.yaml` declares each stage's code, inputs, parameters and outputs. `dvc.lock` stores the md5 of every output, so each run's `dvc.lock` is one exact data version; the data itself is in the DVC remote `gs://ctt-mlops-project-data/dvc-store`.
- There are no separate `.dvc` files: every dataset (the raw snapshot, the features, the text, the embeddings, the reports) is the output of a stage in `dvc.yaml`, and DVC tracks stage outputs in `dvc.lock` instead of `.dvc` files.
- **Data history in Git:** `Data-Pipeline/dvc.lock` is committed. After a cloud run, `./composer/deploy.sh lock` copies that run's `dvc.lock` into the repository, to commit with the code that produced it. The committed one is the Composer run of 2026-10-26 (all nine stages, including the embeddings). `git log -p Data-Pipeline/dvc.lock` shows how the data changed. To restore a version: `git checkout <commit> -- Data-Pipeline/dvc.lock`, then `dvc pull` in `Data-Pipeline` (needs read access to the bucket).
- Every run writes its `dvc.lock` and a version record to `gs://ctt-mlops-project-data/data-pipeline/state/versions/<snapshot>/`, so the exact data of any run can be pulled again.
- The test snapshot is seeded, so the same parameters give the same data and the same `dvc.lock` hashes.
- Every run's `dvc.lock` and version record also go to the state folder of a local run (`Data-Pipeline/local/output/state/versions/`), and its data to the local DVC remote.
- Rebuilding from scratch: on a machine with Docker, `./local/pipeline.sh up` and `trigger`. On Google Cloud, `./composer/deploy.sh setup`, `image`, `gpu-image`, `up` and `trigger` in any project.

## Tests

```bash
pip install -r requirements.txt -r requirements-bias.txt -r Data-Pipeline/requirements.txt -r requirements-dev.txt
pytest                          # src/ code
cd Data-Pipeline && pytest      # pipeline stages, DAG and DVC runner (DAG tests need Airflow)
```

Google Cloud is faked in the tests, so they need no credentials.

## Component details (data acquisition and preprocessing)

Each module below is called by one pipeline stage (`Data-Pipeline/scripts/`), logs what it does, and raises a typed error from `src/error_handling.py` when it fails.

### 1. Data acquisition

`src/ingestion/fetch_aact.py` (stage `acquire`) saves one AACT snapshot, keeping only the 12 tables the project uses, and writes a `manifest.json` (source, snapshot date, file sizes); DVC then stores it in the bucket. Both sources need a free [AACT account](https://aact.ctti-clinicaltrials.org/):

- `SOURCE=db`: the AACT cloud database, with `AACT_USER` and `AACT_PASSWORD` from `Data-Pipeline/.env` (locally) or Composer's secret (made from that file).
- `SOURCE=zip ZIP=gs://...`: the monthly flat-file export, from the bucket or its URL.

Each table is checked for an `nct_id` column and at least one row. Downloads retry up to 3 times and never leave a partial file.

### 2. Text preprocessing

`src/features/text_cleaning.py` cleans the free-text fields: it turns AACT's `~` back into line breaks, removes markdown escapes (`\>=` becomes `>=`), normalizes Unicode (ligatures, full-width characters), collapses whitespace and drops boilerplate lines. It also splits `eligibilities.criteria` into inclusion and exclusion items, handling header variants such as "Key Inclusion Criteria" and "Non-inclusion criteria".

`src/features/build_text_sample.py` (stage `text`) builds one row of cleaned text per trial: title, brief summary, detailed description, inclusion criteria, exclusion criteria and planned outcomes (from `design_outcomes`, which are written at registration, not results).

### 3. Structured preprocessing and features

`src/features/structured_features.py` (stage `preprocess`) cleans the structured fields and builds 155 features per trial: phase and study type, design (allocation, masking, purpose), arms, age range, gender, inclusion and exclusion counts, lead sponsor class and collaborators, sites and countries, intervention types, conditions and MeSH therapeutic areas.

- **Label:** from `overall_status`: 1 for terminated, withdrawn or suspended, 0 for completed, and empty for trials still running (left out of training).
- **Encoding:** categoricals are one-hot encoded against fixed category lists, so every snapshot gets the same columns, with `_unlisted` (a value outside the list) and `_missing` columns. The catch-all is not called `_other` because several AACT fields have a real `OTHER` value. Numeric fields get a `_missing` flag and are filled with 0, so nothing is fitted on the full data.
- **Leakage:** fields filled in after a trial ends are excluded: actual enrollment, completion dates, start date type, `why_stopped` and results. `n_facilities`, `has_us_facility`, `has_single_facility`, `n_countries` and the `_missing` flags of the facility columns are kept but flagged as a leakage risk, because sites are often removed from withdrawn trials. Both lists are in the output's `.meta.json`.

Output: `data/processed/structured_features.csv`, with `nct_id`, `overall_status`, `label`, `start_date`, `study_first_submitted_date` and the features, and `structured_features.meta.json` with the feature list and label counts.

### 4. BioBERT embeddings

`src/features/biobert_embeddings.py` (stage `embeddings`) turns text into 768-number vectors with BioBERT (`dmis-lab/biobert-base-cased-v1.2`, not fine-tuned). Text longer than BioBERT's 512-token limit is split into overlapping 512-token windows (64 tokens of overlap). Each window is mean-pooled, and the windows are averaged, weighted by length, so no text is cut off. Empty text gets a zero vector. If the GPU runs out of memory, the batch is split in half and retried.

The `embeddings` stage runs it on every trial as a Vertex AI custom job on a T4 GPU, in the image `docker/Dockerfile.gpu` (`Data-Pipeline/scripts/embeddings.py`).


### 5. Bias detection and mitigation

`src/bias/` (stage `bias`) slices the finished trials by phase, study type, sponsor class, region, therapeutic area, gender, age group and start era, flags groups that are too small or whose stopped rate is far from the overall rate, and reweights or resamples under-represented groups, on the versioned snapshot.

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
