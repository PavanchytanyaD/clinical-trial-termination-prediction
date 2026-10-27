# Predicting Early Termination of Clinical Trials

MLOps course project (Northeastern University). The system predicts the probability that a registered clinical trial will be stopped early (terminated, withdrawn, or suspended) from its design, and an agentic AI system explains why a flagged trial is at risk by citing similar past trials that were stopped.

## Approach

- **Data:** ClinicalTrials.gov, accessed through the AACT database (monthly PostgreSQL snapshot). One row is one trial, identified by its NCT ID.
- **Baseline:** a classical ML model (gradient-boosted trees or logistic regression) on the structured design features.
- **Main model:** BioBERT text embeddings (not fine-tuned) combined with the structured features in a neural network with a classification head.
- **Agentic system:** looks up similar past stopped trials in the structured data and a text index, reads their recorded stop reasons, and writes a cited explanation.
- **Platform:** Google Cloud (Vertex AI Pipelines, BigQuery, Cloud Storage, Artifact Registry, GKE, Vertex AI Vector Search), with GitHub Actions, DVC, and MLflow.

## Data pipeline (assignment 2)

The data pipeline fetches an AACT snapshot, builds the features and the BioBERT text embeddings, checks the data, checks bias in the data and in a baseline model, and saves a versioned copy of everything. Airflow orchestrates it, DVC versions it, and it runs once a month by default ([the schedule can be changed](#changing-how-often-it-runs)).

```
               +--> preprocess --+--> schema_stats --> validate --+
               |                 +--> anomalies -------------------+
               |                 +--> bias_model ------------------+
    acquire ---+--> bias -------------^                            +--> publish
               +--> text --> embeddings (BioBERT) -----------------+
```

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
| Bias check on a baseline model | `Data-Pipeline/scripts/bias_model.py`, `docs/bias_report.md` section 7 | Pavanchytanya |
| Error handling, code style, contributing | `src/error_handling.py`, `.flake8`, `CONTRIBUTING.md` | Sathyasai |

Every stage is explained in detail in [Data-Pipeline/README.md](Data-Pipeline/README.md).

## Code structure

```
clinical-trial-termination-prediction/
|- .github/workflows/ci.yml  CI/CD: tests on every push, then images -> Artifact Registry
|- Data-Pipeline/            the data pipeline (folder layout from the assignment guidelines)
|  |- dags/                  Airflow DAGs
|  |- scripts/               one module per stage, plus tracking, alerts and the DVC stage runner
|  |- tests/                 tests for the pipeline stages, DAG and DVC runner
|  |- data/, logs/           written by each run (data versioned by DVC, not Git)
|  |- schema/baseline/       how the validation baseline is frozen
|  |- docs/                  pipeline optimization note and Gantt charts
|  |- local/                 way 1: Airflow in Docker on one machine (pipeline.sh)
|  |- composer/              way 2: Cloud Composer on Google Cloud (deploy.sh)
|  |- docker/                pipeline image (Composer pods), BioBERT GPU image (Vertex AI)
|  |- dvc.yaml, params.yaml  DVC stages and their parameters
|  |- dvc.lock               the current data version (hash of every output)
|  \- README.md
|- src/                      shared code the pipeline stages call
|  |- ingestion/             AACT download / database export and table checks
|  |- features/              text cleaning, structured features, BioBERT embeddings
|  |- bias/                  data slicing, bias reports, mitigation, per-group model metrics
|  |- error_handling.py      typed pipeline errors and stage logging
|  \- logging_utils.py       console and file logging for src scripts
|- configs/bias_config.yaml  bias slicing and mitigation settings
|- tests/                    tests for src/
|- docs/                     setup guide, bias report, screenshots of our runs, scoping document
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

## How to run the pipeline: two ways

The same code and the same Airflow DAG run in two places. Pick one:

| | Way 1: on your machine | Way 2: on Google Cloud |
|---|---|---|
| Airflow | in Docker (scheduler, web server, Postgres) | Cloud Composer 3 |
| Each task runs | in a fresh copy of the code in the container | as a pod from the pipeline image |
| BioBERT embeddings | on your CPU | Vertex AI job on a T4 GPU |
| Data versions (DVC remote) and run state | folders in `Data-Pipeline/local/output/` | a Cloud Storage bucket |
| You need | Docker | a Google Cloud project with billing, and the gcloud CLI |
| Cost | free | Composer is billed per hour while it exists |
| Time to the first result | about 15 minutes (image build, then a 2-minute run) | about 1 hour (image builds, Composer creation, run) |

Both run on a seeded test snapshot in the AACT format by default, so no AACT account is needed. With AACT credentials in `Data-Pipeline/.env` ([docs/SETUP.md](docs/SETUP.md)), both can run on the live AACT database (`SOURCE=db`).

### Way 1: on your machine (Docker)

**Requirements:** Git, Bash (on Windows: WSL or Git Bash) and Docker with Compose v2 ([Docker Desktop](https://www.docker.com/products/docker-desktop/) on macOS or Windows, Docker Engine on Linux), with about 4 GB of memory and 6 GB of disk for Docker, and port 8080 free. Nothing else: the image installs every Python dependency and the BioBERT weights.

| Step | Command (from the repository) | What it does |
|---|---|---|
| 1 | `git clone https://github.com/PavanchytanyaD/clinical-trial-termination-prediction.git`<br>`cd clinical-trial-termination-prediction/Data-Pipeline` | Gets the code. |
| 2 | `./local/pipeline.sh up` | Builds the image the first time (Airflow 2.10.5, the pipeline's libraries, CPU-only torch and BioBERT; downloads about 3 GB), starts Postgres, the Airflow scheduler and the web server, and waits until both DAGs are loaded. Prints the UI address. |
| 3 | open <http://localhost:8080>, user `admin`, password `admin` | The Airflow UI: the DAGs `ctt_data_pipeline` and `ctt_freeze_baseline`. |
| 4 | `./local/pipeline.sh trigger` | Starts one run of `ctt_data_pipeline` on a 500-trial test snapshot, waits until it ends (about 2 minutes), and prints the state of every task and where the reports are. Follow it in the UI (Graph, Gantt, Logs). |
| 5 | `ls local/output/state/runs/` | The run's parameters, logs and reports, one folder per task (for example `validate/reports/validation/validation.md`). The data versions are in `local/output/dvc-store/`. |
| 6 | `REASON="first run" ./local/pipeline.sh freeze` | Makes this run's schema and statistics the validation baseline; every later run is validated against it. |
| 7 | `./local/pipeline.sh trigger` again | A second run: DVC skips the stages whose code, data and parameters did not change, and `validate` checks the data against the baseline. |
| 8 | `./local/pipeline.sh down` | Stops Airflow. `local/output/` and the run history stay; `clean` deletes them too. |

Options: `SAMPLE_TRIALS=2000 ./local/pipeline.sh trigger` (a bigger test snapshot), `SNAPSHOT=20261101 ./local/pipeline.sh trigger` (another snapshot date), `SOURCE=db ./local/pipeline.sh trigger` (the live AACT database), `CTT_UI_PORT=8081 ./local/pipeline.sh up` (if port 8080 is taken).

This is step 4 in our test of 2026-10-27 (UI on port 8081, because another program used 8080): all ten tasks succeeded in 2 minutes 5 seconds.

```
$ ./local/pipeline.sh trigger
started ctt_data_pipeline run local__20261027T024530; follow it at http://localhost:8081
dag_id            | execution_date            | task_id      | state   | start_date                       | end_date
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | acquire      | success | 2026-10-27T02:45:34.050163+00:00 | 2026-10-27T02:45:36.959945+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | text         | success | 2026-10-27T02:45:38.103299+00:00 | 2026-10-27T02:45:44.183714+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | preprocess   | success | 2026-10-27T02:45:38.002411+00:00 | 2026-10-27T02:45:44.316001+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | bias         | success | 2026-10-27T02:45:38.045454+00:00 | 2026-10-27T02:45:44.829778+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | anomalies    | success | 2026-10-27T02:45:45.926004+00:00 | 2026-10-27T02:45:56.481892+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | schema_stats | success | 2026-10-27T02:45:45.957692+00:00 | 2026-10-27T02:45:57.434567+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | bias_model   | success | 2026-10-27T02:45:58.293475+00:00 | 2026-10-27T02:46:16.678938+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | validate     | success | 2026-10-27T02:45:58.324913+00:00 | 2026-10-27T02:46:19.482924+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | embeddings   | success | 2026-10-27T02:45:45.906417+00:00 | 2026-10-27T02:47:25.412588+00:00
ctt_data_pipeline | 2026-10-27T02:45:33+00:00 | publish      | success | 2026-10-27T02:47:25.718909+00:00 | 2026-10-27T02:47:38.828531+00:00
ctt_data_pipeline local__20261027T024530: success
reports and logs: .../Data-Pipeline/local/output/state/runs/local__20261027T024530/
```

The run in the Airflow UI (Graph and Gantt views):

![Local run: all ten tasks succeeded](docs/screenshots/local/local_2_graph.png)

![Local run: Gantt view](docs/screenshots/local/local_3_gantt.png)

More details (what each piece does, troubleshooting): [Data-Pipeline/local/README.md](Data-Pipeline/local/README.md).

### Way 2: on Google Cloud

#### 2.1 What we deployed

We run the pipeline in our Google Cloud project. Everything below was created by the commands in 2.3, not by hand:

| Resource | Our value |
|---|---|
| Project | `ctt-mlops` (project ID `ctt-mlops-project`), region `us-central1` |
| Cloud Storage bucket | `gs://ctt-mlops-project-data`: `dvc-store/` (DVC remote: every data version), `data-pipeline/state/` (each run's parameters, `dvc.lock`, logs, reports, the validation baseline and the data-version records), `composer-snapshots/` (Airflow's history between sessions) |
| Artifact Registry | `us-central1-docker.pkg.dev/ctt-mlops-project/ctt-images/`: `data-pipeline` (every Airflow task), `biobert-gpu` (the embeddings job) |
| Airflow | Cloud Composer 3 environment `ctt-composer` (Airflow 2.10.5, size small): each task is a `KubernetesPodOperator` pod from `data-pipeline` |
| GPU step | Vertex AI custom job `ctt-biobert-<run>`: `n1-standard-8` with one NVIDIA T4, image `biobert-gpu` |
| Identities | `pipeline-sa` (Composer, its pods, the Vertex AI job), `cicd-sa` (image builds) |
| CI/CD | GitHub Actions (`.github/workflows/ci.yml`): style check and all tests on every push; on `main` it then builds both images with Cloud Build and uploads the DAG if Composer is up. It signs in through Workload Identity Federation, so no key exists. |

Composer is billed for every hour it exists and cannot be paused, so we create it for a working session and delete it afterwards (`up` / `down`). The data, the run history and the images stay. It is not running now; the screenshots in 2.5 show it running.

#### 2.2 Requirements in your Google Cloud project

- A project with **billing enabled**, where you are **Owner** (setup creates service accounts and grants roles). A new project keeps everything separate and is easy to delete.
- The **[Google Cloud CLI](https://cloud.google.com/sdk/docs/install)** (`gcloud`), Git and Bash. [Cloud Shell](https://shell.cloud.google.com) has all three.
- **GPU quota** for Vertex AI custom training: one NVIDIA T4 in your region (IAM & Admin > Quotas, "Custom model training Nvidia T4 GPUs"). New projects sometimes have 0; without it, set `accelerator: ""` in the `embeddings` section of `Data-Pipeline/params.yaml` and commit it, and the job runs on CPU.
- Nothing else to enable by hand: `setup` enables the APIs (Composer, Artifact Registry, Cloud Build, Vertex AI, IAM, Storage).
- Optional: `Data-Pipeline/.env` with AACT credentials and SMTP settings for alert emails ([docs/SETUP.md](docs/SETUP.md)); `up` stores them as a Composer secret.

#### 2.3 Commands to deploy it in your project

| Step | Command (from `Data-Pipeline/`) | What it does | Time |
|---|---|---|---|
| 0 | `gcloud auth login`<br>`gcloud config set project YOUR_PROJECT_ID`<br>`export PROJECT=YOUR_PROJECT_ID` | Signs in and picks the project. Every name below follows from `PROJECT`: the bucket `${PROJECT}-data`, the images, the service accounts (`REGION`, `BUCKET`, `ENV_NAME` can be overridden). | 1 min |
| 1 | `./composer/deploy.sh setup` | Once per project: enables the APIs, creates the bucket and the service accounts `pipeline-sa` and `cicd-sa`, and gives them their roles. Safe to re-run. | 2 min |
| 2 | `./composer/deploy.sh image`<br>`./composer/deploy.sh gpu-image` | Builds the two images with Cloud Build from the committed code and pushes them to Artifact Registry. | 15 min |
| 3 | `./composer/deploy.sh up` | Creates the Composer environment as `pipeline-sa`, sets the DAG's settings (images, bucket, state folder, DVC remote) and the secret, uploads the DAGs, and prints the Airflow UI address. | 25 min |
| 4 | `./composer/deploy.sh trigger` | Starts one run on the test snapshot (`SNAPSHOT=`, `SOURCE=db`, `SAMPLE_TRIALS=` to choose). Follow it in the UI. | 25 min |
| 5 | `./composer/deploy.sh ui`<br>`gcloud storage ls -r gs://${PROJECT}-data/data-pipeline/state/` | The UI address; the run's logs and reports in `state/runs/<run_id>/`, its data version in `state/versions/<snapshot>/`. | |
| 6 | `REASON="first run" ./composer/deploy.sh freeze` | Makes the run's schema and statistics the validation baseline. | 2 min |
| 7 | `./composer/deploy.sh lock`, then `git commit Data-Pipeline/dvc.lock` | Copies the run's `dvc.lock` into the repository, so Git keeps the history of the data versions. | |
| 8 | `./composer/deploy.sh down` | Saves Airflow's history to the bucket and deletes Composer: **billing stops**. The next `up` restores the history. | 10 min |

To remove everything afterwards: `gcloud projects delete ${PROJECT}`.

#### 2.4 What we ran on our project

The session of 2026-10-26 on `ctt-mlops-project`, with the outputs (shortened):

```
$ cd Data-Pipeline
$ gcloud config set project ctt-mlops-project
$ ./composer/deploy.sh setup                        # once per project

# The images: built by GitHub Actions on every push (screenshot 1 in 2.5);
# by hand it is ./composer/deploy.sh image && ./composer/deploy.sh gpu-image

$ ./composer/deploy.sh up
Waiting for [projects/ctt-mlops-project/locations/us-central1/environments/ctt-composer] to be created ...
https://f3c7d1003b414cc6bd3aa7d79c31315a-dot-us-central1.composer.googleusercontent.com

$ SNAPSHOT=20261026 ./composer/deploy.sh trigger
Executing the command: [ airflow dags trigger ctt_data_pipeline -c {"snapshot": "20261026", "source": "sample", "zip": "", "sample_trials": ""} ]...
Command has been started.

$ ./composer/deploy.sh down
snapshotPath: gs://ctt-mlops-project-data/composer-snapshots/ctt-mlops-project_us-central1_ctt-composer_2026-10-26T16-18-26
Delete in progress for environment [projects/ctt-mlops-project/locations/us-central1/environments/ctt-composer] ...
```

The run took 24 minutes and all tasks succeeded. The data version of this session is the `dvc.lock` committed in this repository.

#### 2.5 Proof that every component ran

The screenshots are from that session (2026-10-26). The DAG had nine tasks then; `bias_model` was added on 2026-10-27 and is tested in the local runs above.

**1. GitHub Actions:** lint, the three test suites, both image builds and the DAG upload passed.

![GitHub Actions run](docs/screenshots/01_github_actions_ci.png)

**2. Artifact Registry and Cloud Build:** the two images in `ctt-images`, built by Cloud Build.

![Artifact Registry: biobert-gpu and data-pipeline](docs/screenshots/02_artifact_registry_images.png)

![Cloud Build history](docs/screenshots/03_cloud_build_history.png)

**3. Cloud Composer:** `up` creating `ctt-composer` (Airflow 2.10.5, us-central1), the environment running as `pipeline-sa`, and the settings `deploy.sh` gave it.

![Composer environment being created](docs/screenshots/07_composer_creating.png)

![Composer environment configuration](docs/screenshots/15_composer_environment_config.png)

![Composer environment variables](docs/screenshots/16_composer_environment_variables.png)

**4. Airflow on Composer:** both DAGs loaded; the run's Graph (every task a `KubernetesPodOperator` pod, all green) and Gantt view. The red bar is the first attempt of `bias`, which failed and succeeded on Airflow's automatic retry.

![Airflow DAG list](docs/screenshots/09_airflow_dags_list.png)

![Airflow graph: all tasks succeeded](docs/screenshots/10_airflow_graph_success.png)

![Airflow Gantt view](docs/screenshots/11_airflow_gantt.png)

**5. A task log:** `embeddings` waiting for the Vertex AI job, then DVC updating `dvc.lock` and pushing the embeddings.

![embeddings task log](docs/screenshots/12_airflow_task_log_embeddings.png)

**6. Vertex AI:** the BioBERT jobs on the T4 (the stopped one was cancelled on purpose while testing that a stopped task cancels its GPU job).

![Vertex AI custom jobs](docs/screenshots/06_vertex_ai_custom_jobs.png)

**7. Cloud Storage:** the bucket with the DVC remote and the run state; the state folder with the shared `dvc.lock`, every run's files (`runs/`) and the data versions (`versions/`).

![Bucket ctt-mlops-project-data](docs/screenshots/04_gcs_bucket.png)

![Run state folder](docs/screenshots/05_gcs_run_state.png)

**8. The runs:** both triggered runs succeeded; the run's timeline from Airflow's task times (1,435 s; `embeddings` includes Vertex AI starting the GPU machine).

![DAG runs](docs/screenshots/14_airflow_dag_runs.png)

![Gantt chart of the Composer run](docs/screenshots/08_composer_run_gantt.png)

More details: [Data-Pipeline/composer/README.md](Data-Pipeline/composer/README.md). CI/CD from a fork: set the repository variables `GCP_PROJECT`, `WIF_PROVIDER` and `CICD_SA` (see the top of `.github/workflows/ci.yml`).

### Changing how often it runs

Airflow's scheduler starts `ctt_data_pipeline` on a schedule. The default is **monthly, on the 5th at 06:00 UTC**, after AACT publishes its monthly export. The schedule is not hard-coded: it is the Airflow Variable `ctt_schedule`, which can be changed while Airflow runs. The scheduler re-reads the DAG file regularly and uses the new schedule within about a minute, with nothing to redeploy:

| | Way 1 (local) | Way 2 (Google Cloud) |
|---|---|---|
| Show the schedule | `./local/pipeline.sh schedule` | `./composer/deploy.sh schedule` |
| Weekly | `./local/pipeline.sh schedule "@weekly"` | `SCHEDULE="@weekly" ./composer/deploy.sh schedule` |
| Any cron, e.g. Mondays 06:00 UTC | `./local/pipeline.sh schedule "0 6 * * 1"` | `SCHEDULE="0 6 * * 1" ./composer/deploy.sh schedule` |
| Manual runs only | `./local/pipeline.sh schedule none` | `SCHEDULE=none ./composer/deploy.sh schedule` |
| Back to monthly | `./local/pipeline.sh schedule default` | `SCHEDULE=default ./composer/deploy.sh schedule` |

We tested it locally: the DAG list showed `@weekly` right after the change, and `default` brought back `0 6 5 * *`, with Airflow running the whole time. Because the DAG has `catchup=False`, a change to a more frequent schedule starts one run for the most recent interval that has already ended (for `@weekly`, last week), then follows the new schedule.

![DAG list after the change: Schedule @weekly](docs/screenshots/local/local_1_dags.png)

The same Variable can be edited in the Airflow UI (Admin > Variables > `ctt_schedule`). If it is not set, the environment variable `CTT_SCHEDULE` is used, and if that is not set either, monthly. Scheduled runs only happen while Airflow is running: locally while the containers are up, on Google Cloud while the Composer environment exists. Because we delete Composer between sessions to save credits, our cloud runs are triggered by hand; keeping `ctt-composer` up would run them on the schedule.

## Data versioning and reproducibility

- `Data-Pipeline/dvc.yaml` declares each stage's code, inputs, parameters and outputs. `dvc.lock` stores the md5 of every output, so each run's `dvc.lock` is one exact data version; the data itself is in the DVC remote `gs://ctt-mlops-project-data/dvc-store`.
- There are no separate `.dvc` files: every dataset (the raw snapshot, the features, the text, the embeddings, the reports) is the output of a stage in `dvc.yaml`, and DVC tracks stage outputs in `dvc.lock` instead of `.dvc` files.
- **Data history in Git:** `Data-Pipeline/dvc.lock` is committed. After a cloud run, `./composer/deploy.sh lock` copies that run's `dvc.lock` into the repository, to commit with the code that produced it. The committed one is from the Composer session of 2026-10-26: every DVC stage of that day's pipeline, including the embeddings (`bias_model` was added a day later, so it is not in it yet). `git log -p Data-Pipeline/dvc.lock` shows how the data changed. To restore a version: `git checkout <commit> -- Data-Pipeline/dvc.lock`, then `dvc pull` in `Data-Pipeline` (needs read access to the bucket).
- Every run writes its `dvc.lock` and a version record to `gs://ctt-mlops-project-data/data-pipeline/state/versions/<snapshot>/`, so the exact data of any run can be pulled again.
- The test snapshot is seeded, so the same parameters give the same data and the same `dvc.lock` hashes.
- Every run's `dvc.lock` and version record also go to the state folder of a local run (`Data-Pipeline/local/output/state/versions/`), and its data to the local DVC remote.
- Rebuilding from scratch: [way 1](#way-1-on-your-machine-docker) on any machine with Docker, or [way 2](#way-2-on-google-cloud) in any Google Cloud project.

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

The `embeddings` stage runs it on every trial: on Google Cloud as a Vertex AI custom job on a T4 GPU, in the image `docker/Dockerfile.gpu`, and locally on the CPU (`Data-Pipeline/scripts/embeddings.py`).


### 5. Bias detection and mitigation

`src/bias/` (stage `bias`) slices the finished trials by phase, study type, sponsor class, region, therapeutic area, gender, age group and start era, flags groups that are too small or whose stopped rate is far from the overall rate, and reweights or resamples under-represented groups, on the versioned snapshot.

`Data-Pipeline/scripts/bias_model.py` (stage `bias_model`) then checks a model: it trains a baseline logistic regression on the structured features, evaluates it per group with Fairlearn, and mitigates the gaps two ways (training with the reweighting weights, and one decision threshold per region). On the real 2019 AACT data, per-region thresholds cut the recall gap between regions from 0.32 to 0.13, at a cost of 0.03 PR-AUC.

Findings, both mitigations and their trade-offs are in [docs/bias_report.md](docs/bias_report.md).

## Team

- Atharva Nilesh Mahadik
- Danush Gopinath
- Nithish Bhat
- Pavanchytanya Dharmapuri
- Sathyasai Rajan
- Sivapriya Sugumaran

## Data rights

ClinicalTrials.gov and AACT (CTTI) data are public. The data is trial-level and contains no patient-level personal data.
