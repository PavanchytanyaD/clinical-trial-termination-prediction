# Data Pipeline

Monthly data pipeline for the clinical-trial early-termination project. It
ties every part of the data pipeline assignment together:

| Assignment item | Where |
|---|---|
| 1. Data acquisition | `scripts/acquire.py` calls `../src/ingestion/fetch_aact.py` |
| 2. Data preprocessing | `scripts/preprocess.py` calls `../src/features/structured_features.py` |
| 3. Test modules | `tests/` here and `../tests/` |
| 4. Pipeline orchestration (Airflow DAGs) | `dags/ctt_data_pipeline.py`, on Cloud Composer (`composer/`) or Airflow in Docker (`local/`) |
| 5. Data versioning with DVC | `dvc.yaml`, `dvc.lock`, `.dvc/config` (GCS remote) |
| 6. Tracking and logging | `scripts/tracking.py`, `scripts/alerts.py` |
| 7. Data schema and statistics generation | `scripts/schema_stats.py`, `scripts/validate.py`, `schema/` |
| 8. Anomaly detection and alerts | `scripts/anomalies.py` |
| 9. Pipeline flow optimization | `docs/pipeline_optimization.md` |
| Bias detection and mitigation | `scripts/bias.py` calls `../src/bias/` |
| BioBERT text embeddings | `scripts/text.py`, `scripts/embeddings.py` call `../src/features/`; GPU job on Vertex AI |

The stages call the team's shared code in `../src/` (ingestion, features,
bias, error handling). The same DAG runs in two places:

- **Google Cloud**, our deployment (project `ctt-mlops-project`). GitHub
  Actions tests every push and builds the images into Artifact Registry.
  Cloud Composer orchestrates and runs each step as a pod from that image.
  Data versions live in the GCS bucket `gs://ctt-mlops-project-data`, and
  BioBERT runs as a Vertex AI GPU job. Anyone can deploy it in their own
  project with `composer/deploy.sh`. Both are shown in
  [../docs/cloud_deployment.md](../docs/cloud_deployment.md), with
  screenshots from a run.
- **One machine**: Airflow in Docker (`local/`). The run state and the DVC
  remote are folders in `local/output/`, and BioBERT runs on the CPU, so no
  cloud account is needed ([local/README.md](local/README.md)).

Without AACT credentials, `acquire` writes a seeded test snapshot in the
AACT format, so the whole pipeline can still be exercised end to end.

## How it fits together

```
   GitHub push -> GitHub Actions: tests, then images -> Artifact Registry
                       |
     Cloud Composer (Airflow): schedule, order, retries, failure email
                       |
               +--> preprocess --+--> schema_stats --> validate --+
               |                 +--> anomalies -------------------+
    acquire ---+--> bias ------------------------------------------+--> publish
               +--> text --> embeddings (Vertex AI T4 GPU job) ----+
                       |
   each task = one pod from the pipeline image, running one DVC stage
   (scripts/run_stage.py)
                       |
        DVC: tracks inputs, params and outputs (dvc.lock),
             skips unchanged stages, stores every data version in GCS;
        run state (params, dvc.lock, logs, reports, baseline) in GCS too
```

| Stage | What it does | Output |
|---|---|---|
| `acquire` | Writes one AACT snapshot (12 tables). Source `db` (AACT cloud PostgreSQL) or `zip` (monthly export) via `src/ingestion/fetch_aact.py`, or `sample` (seeded test data) | `data/raw/aact_snapshot/` |
| `preprocess` | One row per trial with the label and features. Calls `src/features/structured_features.py` when present, else a placeholder | `data/processed/structured_features.csv` |
| `bias` | Slices the finished trials (phase, sponsor, region, area, gender, age group, era), flags under-represented groups and rate gaps, reweights them (`src/bias/`) | `reports/bias/`, `data/processed/trial_slices.csv` |
| `schema_stats` | Per-column statistics, and a Great Expectations suite (the schema) inferred from the snapshot | `reports/statistics/`, `schema/generated/` |
| `anomalies` | Flags duplicate or malformed ids, invalid labels, impossible values (critical) and missing-value spikes, outliers, bad dates (warnings); emails an alert, stops the run on critical findings | `reports/anomalies/` |
| `validate` | Checks the snapshot against the frozen baseline schema and statistics; stops the run on critical problems and emails an alert | `reports/validation/` |
| `text` | Cleaned registration-time text of every trial (title, summary, description, inclusion, exclusion, outcomes) via `src/features/build_text_sample.py` | `data/processed/text_all.jsonl` |
| `embeddings` | BioBERT embeddings per text field (`src/features/biobert_embeddings.py`), on a Vertex AI GPU by default; see below | `data/processed/embeddings/` |
| `publish` | Pushes every output to the DVC remote and writes a data-version record | `logs/versions/`, GCS `state/versions/` |

## Folder structure

```
Data-Pipeline/
|- dags/ctt_data_pipeline.py   Airflow DAGs (each task a Composer pod)
|- scripts/                    pipeline steps, run as python -m scripts.<step>
|  |- acquire.py               stage 1 adapter (ingestion code or placeholder)
|  |- sample_snapshot.py       seeded test snapshot in the AACT format
|  |- preprocess.py            stage 2 adapter (feature code or placeholder)
|  |- bias.py                  bias detection and mitigation on the snapshot
|  |- text.py                  cleaned trial text (BioBERT input)
|  |- embeddings.py            BioBERT embeddings (Vertex AI GPU job)
|  |- schema_stats.py          statistics + Great Expectations schema
|  |- anomalies.py             anomaly detection + alerts
|  |- validate.py              schema validation + drift checks + alerts
|  |- run_stage.py             runs one DVC stage with its state in GCS (each task)
|  |- tracking.py              logging and per-step run records (+ MLflow)
|  |- alerts.py                SMTP email alerts
|  |- config.py                paths and params
|- schema/baseline/            how the baseline (kept in GCS) is frozen
|- tests/                      unit tests (pytest)
|- docs/                       pipeline optimization note and Gantt charts
|- docker/Dockerfile           image the Composer tasks run
|- docker/Dockerfile.gpu       BioBERT image the Vertex AI GPU job runs
|- composer/                   Cloud Composer: setup, up, trigger, down (deploy.sh)
|- local/                      Airflow in Docker on one machine (pipeline.sh,
|                              docker-compose.yaml, run_task.sh)
|- dvc.yaml, params.yaml       DVC stages and their parameters
|- dvc.lock                    hash of every output: the current data version
|- requirements.txt, .env.example
```

`data/`, `reports/`, `schema/generated/` and `logs/` are produced inside each
task and never committed. DVC stores the data in the remote, and the run
state folder keeps the logs, reports and the latest `dvc.lock`. On Google
Cloud both are in the GCS bucket; locally they are in `local/output/`.

## Run it on your machine

```bash
./local/pipeline.sh up        # Airflow in Docker at http://localhost:8080
./local/pipeline.sh trigger   # one run on the test snapshot; waits for it
./local/pipeline.sh down
```

Details: [local/README.md](local/README.md).

## Run it on Google Cloud

| | Where | Use |
|---|---|---|
| Tests and image builds | GitHub Actions (`../.github/workflows/ci.yml`) + Cloud Build | every push |
| Images | Artifact Registry `ctt-images` | `data-pipeline` (Composer pods), `biobert-gpu` (Vertex AI) |
| Airflow | Cloud Composer 3 `ctt-composer` | created for a working session, deleted after (billed per hour, cannot be paused) |
| Each step | a pod on Composer's cluster | the pipeline image, one DVC stage |
| BioBERT | Vertex AI custom job, `n1-standard-8` + T4 | the `embeddings` stage |
| Data versions | `gs://ctt-mlops-project-data/dvc-store` (DVC remote) | every output of every run |
| Run state | `gs://ctt-mlops-project-data/data-pipeline/state` | params, `dvc.lock`, logs, reports, baseline, version records |
| Airflow history | `gs://ctt-mlops-project-data/composer-snapshots` | saved by `down`, restored by `up` |

Composer and the pods run as the service account `pipeline-sa`; CI signs in
as `cicd-sa` through Workload Identity Federation (no key).

**Code changes:** push to GitHub. CI runs the style check and the tests
(`src/`, the stages and the DAGs under Airflow 2.10.5) and, on `main` and
`pavan/integration`, builds both images tagged `:latest` and with the commit.
If Composer is up, it also uploads the new DAG.

**A working session** (a Google account with access to `ctt-mlops-project`):

```bash
gcloud auth login && gcloud config set project ctt-mlops-project
./composer/deploy.sh setup     # once per project
./composer/deploy.sh up        # about 25 minutes; prints the Airflow UI address
./composer/deploy.sh trigger   # or SOURCE=db SNAPSHOT=20261101 ./composer/deploy.sh trigger
./composer/deploy.sh down      # snapshot Airflow's history, delete Composer
```

Details: [composer/README.md](composer/README.md). To deploy it in your own
Google Cloud project, set `PROJECT` and follow
[../docs/cloud_deployment.md](../docs/cloud_deployment.md#2-deploy-it-in-your-own-project).

**Real AACT data:** put `AACT_USER` and `AACT_PASSWORD` in
`Data-Pipeline/.env` (from `.env.example`); `up` stores them as a Composer
secret the pods read. Trigger with `SOURCE=db`, or with
`SOURCE=zip ZIP=gs://...` for a monthly export in the bucket.

## BioBERT embeddings (Vertex AI)

`text` writes the cleaned text of every trial; `embeddings` turns each text
field into a 768-dim BioBERT vector (`dmis-lab/biobert-base-cased-v1.2`,
used as is; long texts are split into overlapping 512-token windows and
averaged). Output in `data/processed/embeddings/`: `ids.txt` (row order),
one `<field>.npy` float16 array per field and `meta.json`. DVC versions it
like every other output and skips the stage when the text did not change.

`scripts/embeddings.py` uploads the text to the bucket, starts a Vertex AI
custom job on an `n1-standard-8` with one T4 GPU running the image
`biobert-gpu` (`docker/Dockerfile.gpu`, model weights included), waits for
it, and downloads the result. If the task is stopped or times out, the job
is cancelled, so no GPU is left running. `params.yaml` (section
`embeddings`) sets the machine and model settings; `max_trials` embeds only
the first N trials for a quick test. In the local Docker Airflow
(`PIPELINE_EMBEDDINGS=local`), the stage runs BioBERT on the CPU in the task
itself instead.

## 4. Pipeline orchestration (Airflow)

`dags/ctt_data_pipeline.py` defines the DAG `ctt_data_pipeline`:

- `acquire` first, then `preprocess`, `bias` and `text` in parallel, then
  `schema_stats`, `anomalies` and `embeddings` in parallel, then
  `validate`, then `publish` (see the diagram above). It runs on the 5th of every month at 06:00 UTC (after the monthly AACT export), or
  triggered by hand with a config such as
  `{"snapshot": "20261001", "source": "zip", "zip": "gs://.../export.zip"}`.
- On Composer, every task is a `KubernetesPodOperator`: a pod from the
  pipeline image (`docker/Dockerfile`) that runs one DVC stage with
  `python -m scripts.run_stage <stage>`. Airflow only orchestrates, and the
  pipeline's libraries never clash with Composer's own Airflow packages.
- In the local Docker Airflow (`CTT_RUNNER=local`), every task is a
  `BashOperator` running `local/run_task.sh <stage>`. It copies the code to
  a fresh folder and runs the same command there, with the pipeline's
  libraries in their own virtualenv.
- A second DAG, `ctt_freeze_baseline`, runs `run_stage freeze` as one task
  (`./composer/deploy.sh freeze` or `./local/pipeline.sh freeze`).
- 2 retries with exponential backoff, a timeout per task and per run,
  `max_active_runs=1` (stages share `dvc.lock`), and an email to
  `CTT_ALERT_EMAILS` when a task fails.
- `validate` and `anomalies` exit with an error on critical data problems,
  so the run stops before `publish` and no bad data version is recorded.
- Parallel tasks are safe with DVC: each task merges only its own stage
  into the shared `dvc.lock` with a conditional write that is retried when
  another task wrote first. In GCS the condition is the object's
  generation; in a local folder it is a file lock and the file's hash.

Because every task starts from a fresh copy of the code, `run_stage.py`
keeps the run's shared files in a state folder (`CTT_STATE_URI`, default
`gs://ctt-mlops-project-data/data-pipeline/state`; `local/output/state` in
the local Docker Airflow): it downloads
`params.yaml`, the latest `dvc.lock` and the baseline and pulls the stage's
inputs before the stage, then pushes the outputs and uploads `dvc.lock`,
logs and reports afterwards (logs and reports also when the stage fails).

## 5. Data versioning with DVC

- DVC is initialised in this folder (`dvc init --subdir`), so it does not
  touch the rest of the repository.
- `dvc.yaml` declares each stage's code, input data, parameters and outputs.
  `dvc.lock` records the md5 hash of every output, so each run's
  `dvc.lock` is one exact data version, and a model can be traced back to the
  snapshot it was trained on.
- The data itself goes to the DVC remote in the GCS bucket
  `gs://ctt-mlops-project-data`; each run's `dvc.lock` is kept in the state
  folder next to it.

### DVC remote

The remote is `gs://ctt-mlops-project-data/dvc-store` (`.dvc/config`). The
Composer pods reach it as `pipeline-sa`, without keys. `PIPELINE_DVC_REMOTE`
replaces it for one run without touching `.dvc/config`. `deploy.sh` sets it
to `gs://<your bucket>/dvc-store` in another project, and the local Docker
Airflow sets it to `local/output/dvc-store`.

There are no separate `.dvc` files: every dataset is the output of a stage
in `dvc.yaml`, and DVC records stage outputs in `dvc.lock`, which plays the
role of `.dvc` files. `dvc.lock` is committed, so Git keeps the history of
the data next to the code. After a cloud run, `./composer/deploy.sh lock`
copies that run's `dvc.lock` here to commit. The committed one is from the
Composer run of 2026-10-26, and `dvc pull` restores all 41 of its files
(including the embeddings) from the bucket. `git log -p dvc.lock` shows how
the data changed; `git checkout <commit> -- dvc.lock` followed by `dvc pull`
brings back an older version.

Every run writes its data version to the state folder:
`state/versions/<snapshot>/<run_id>.json` (output hashes and the validation
summary) and `state/versions/<snapshot>/dvc.lock`. Restoring a version is a
`dvc pull` with that `dvc.lock`.

## 6. Tracking and logging

- **Logs:** every step logs to the console and to `logs/<step>.log`. The
  console output appears in the Airflow task log (and in Cloud Logging on
  Composer), and the log files are uploaded to the run's state folder.
- **Run records:** every step appends a JSON line to
  `logs/runs/<run_id>.jsonl` with start and end time, duration, status, error,
  git commit, metrics (rows, label counts, checks passed) and an md5
  fingerprint of each output. The run id is the Airflow `run_id`, so one file
  shows every step of one DAG run.
- **MLflow (optional):** set `MLFLOW_TRACKING_URI` and install `mlflow`, and
  each step's metrics are also logged to the experiment `ctt-data-pipeline`.
- **Alerts:** email over SMTP (`SMTP_*`, `ALERT_EMAIL_*` in `.env` or the
  environment). Validation problems send a report of the failed checks. A
  step that crashes outside Airflow sends the traceback; under Airflow the
  failure email comes from Airflow instead, so nobody gets two. Without SMTP
  settings the alert is only logged.
- **Data versions:** `publish` writes `state/versions/<snapshot>/<run_id>.json`
  (output hashes plus the validation summary).

## 7. Data schema and statistics generation

`schema_stats` generates, for every snapshot:

- **Statistics** (`reports/statistics/statistics.json` and `.md`): rows,
  columns, label counts and stopped-trial rate; per column: type, missing
  count and rate, distinct values, min, max, mean, std and quantiles for
  numbers, top values for categories, invalid-format rate for dates and trial
  ids.
- **Schema** (`schema/generated/expectation_suite.json`): a Great Expectations
  suite inferred from the snapshot. Each expectation has a severity:

| Severity | Expectations | Meaning |
|---|---|---|
| critical | exact column set; minimum rows; `nct_id` not null, unique, `NCT` + 8 digits; `label` in {0, 1}; features numeric; no nulls in features that had none; flags and one-hot columns in {0, 1}; `overall_status` not null | the data is broken: the run stops |
| warning | numeric values in the reference range (with slack); categories in the known set; date format; missing rate no higher than the reference plus 5 points | the data looks different: someone should check |

Data quality is validated over time: `validate` checks each new snapshot
against a **baseline** frozen from a trusted snapshot, not against itself:

1. the baseline Great Expectations suite (`schema/baseline/`),
2. drift against the baseline statistics: row count change, stopped-trial
   rate change, missing-rate increase per column, and mean shift per numeric
   column in baseline standard deviations (thresholds in `params.yaml`,
   section `validate`).

Results go to `reports/validation/validation.json` (every check),
`validation.md` (failed checks) and `summary.json` (DVC metrics). With
`validate.fail_on: critical` a critical failure fails the stage (exit code
1); `alert_on: warning` emails warnings too.

Freeze a baseline after a run on a snapshot you trust:

```bash
REASON="first real AACT snapshot" ./composer/deploy.sh freeze
```

It is stored in `state/baseline/` in the state folder (`./local/pipeline.sh
freeze` locally), and every task downloads it before running, so all later
runs validate against it.

Until a baseline is frozen, `validate` checks the snapshot against its own
suite and says so in the report.

## 8. Anomaly detection and alerts

`scripts/anomalies.py` checks each snapshot's feature table for values that
are wrong or suspicious on their own (`validate` compares with the frozen
baseline instead):

| Severity | Check |
|---|---|
| critical | duplicate trial ids; ids that are not `NCT` + 8 digits; labels other than 0, 1 or empty; negative counts; ages outside 0-120 or a minimum above the maximum; start or registration years outside 1900 to next year; every finished trial in one class |
| warning | a column missing for more than 30% of trials; more than 5% outliers in a column (outside Q1 - 3 IQR, Q3 + 3 IQR); dates not in `YYYY-MM` or `YYYY-MM-DD` |

Values filled in during preprocessing (their `_missing` flag is set) are not
counted as impossible. Thresholds and the fail and alert policies are in
`params.yaml` (section `anomalies`). Findings go to
`reports/anomalies/anomalies.md`, `.json` and `summary.json`, and an email
alert lists them. The checks already caught one real problem: placeholder
trials whose minimum age was above their maximum age.

## Bias stage

`scripts/bias.py` runs the team's bias code (`../src/bias/`, settings in
`../configs/bias_config.yaml`) on the same snapshot as the rest of the run,
through the `snapshot` source of `src/bias/load_data.py`, so no AACT
database account is needed. Outputs: `reports/bias/data_slices.csv`,
`data_summary.json` and `mitigation/`. The findings and trade-offs are
discussed in `../docs/bias_report.md`.

## 9. Pipeline flow optimization

The Airflow Gantt chart of a sequential run showed every task waiting for the
previous one, although `bias` and `text` need only the raw snapshot and
`schema_stats` and `anomalies` need only the features. Running them in
parallel, so all of that work now happens while the Vertex AI GPU job runs,
cut a 50,000-trial run from 702.5 s to 607.5 s. Both Gantt charts
and how to reproduce them: `docs/pipeline_optimization.md`.

## Interface for the acquisition and preprocessing code

The adapters only need this from the other workstreams' code:

- **Acquisition** (`src/ingestion/fetch_aact.py`): `download(url, dest)`,
  `extract_tables(zip_path, out_dir)`, `export_from_db(out_dir)` and
  `validate(out_dir)`, writing one pipe-delimited `<table>.txt` per table in
  `REQUIRED_TABLES` (`scripts/config.py`), each with an `nct_id` column.
- **Preprocessing** (`src/features/structured_features.py`):
  `build_features(export_dir)` returning `(DataFrame, meta)`, one row per
  `nct_id`, with columns `nct_id`, `overall_status`, `label` (1 stopped, 0
  completed, empty while running), `start_date`,
  `study_first_submitted_date`, then numeric features.

With that code, `preprocess` builds 155 features and `schema_stats`
generates 475 expectations, all passing on the sample snapshot. Failure
points raise the typed errors from `../src/error_handling.py`
(`DataUnavailableError`, `DataValidationError`, `ConfigError`), so run
records and alert emails say what kind of failure happened.

## Tests

```bash
pip install -r requirements.txt && pytest
```

78 tests cover the test snapshot and features, statistics values, the
inferred schema, every critical and warning check (a dropped column,
duplicate or malformed ids, an invalid label, out-of-range values), drift
detection, the snapshot date in the statistics, every anomaly check, the
bias stage, the text stage, the Vertex AI
job (spec, CPU-only spec, polling, cancel on stop, the round trip through the
bucket, settings from the environment), the local BioBERT backend, the fail
and alert policies, run records, email alerts (with a fake SMTP server),
parameter overrides, the `dvc.lock` merge under concurrent writes (GCS and
a local state folder), the DVC remote override, the stage graph, and the DAG
on both runners (the 5 DAG tests need Airflow installed and are skipped
otherwise). Google Cloud is faked in the tests, so they need no credentials.

## Reproducibility

- **Any machine with Docker:** `./local/pipeline.sh up` and `trigger`
  rebuild and run the whole pipeline with no account
  ([local/README.md](local/README.md)).
- **Any Google Cloud project:** `PROJECT=<id> ./composer/deploy.sh setup`,
  `image`, `gpu-image`, `up`, `trigger`
  ([../docs/cloud_deployment.md](../docs/cloud_deployment.md)).
- The test snapshot is seeded, so the same parameters give the same files
  and the same `dvc.lock` hashes in both places. Every run's `dvc.lock`
  stays in `state/versions/`, so the exact data of any run can be pulled
  again from the DVC remote.
