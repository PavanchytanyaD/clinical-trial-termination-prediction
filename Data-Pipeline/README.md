# Data Pipeline

Monthly data pipeline for the clinical-trial early-termination project. It
covers four parts of the data pipeline assignment:

| Assignment item | Where |
|---|---|
| 4. Pipeline orchestration (Airflow DAGs) | `dags/ctt_data_pipeline.py`, runs on Cloud Composer |
| 5. Data versioning with DVC | `dvc.yaml`, `dvc.lock`, `.dvc/config` (GCS remote) |
| 6. Tracking and logging | `scripts/tracking.py`, `scripts/alerts.py` |
| 7. Data schema and statistics generation | `scripts/schema_stats.py`, `scripts/validate.py`, `schema/` |

Data acquisition (item 1) and preprocessing (item 2) belong to other
workstreams. The pipeline calls them through two small adapters
(`scripts/acquire.py`, `scripts/preprocess.py`). Until that code is merged,
the adapters fall back to placeholders, so the whole pipeline already runs
end to end.

## How it fits together

```
           Cloud Composer (Airflow)  -- schedule, order, retries, failure email
                       |
   acquire -> preprocess -> schema_stats -> validate -> publish
      |           |             |              |           |
      +-----------+------ each task = one DVC stage -------+
                       |
        DVC: tracks inputs, params and outputs (dvc.lock),
             skips unchanged stages, stores every data version in GCS
```

| Stage | What it does | Output |
|---|---|---|
| `acquire` | Writes one AACT snapshot (12 tables). Source `sample` (placeholder), `zip` or `db` (calls `src/ingestion/fetch_aact.py`) | `data/raw/aact_snapshot/` |
| `preprocess` | One row per trial with the label and features. Calls `src/features/structured_features.py` when present, else a placeholder | `data/processed/structured_features.csv` |
| `schema_stats` | Per-column statistics, and a Great Expectations suite (the schema) inferred from the snapshot | `reports/statistics/`, `schema/generated/` |
| `validate` | Checks the snapshot against the frozen baseline schema and statistics; stops the run on critical problems and emails an alert | `reports/validation/` |
| `publish` | Pushes every output to the DVC remote and writes a data-version record | `logs/versions/`, GCS `state/versions/` |

## Folder structure

```
Data-Pipeline/
|- dags/ctt_data_pipeline.py   Airflow DAG (Cloud Composer, or local for testing)
|- scripts/                    pipeline steps, run as python -m scripts.<step>
|  |- acquire.py               stage 1 adapter (ingestion code or placeholder)
|  |- sample_snapshot.py       placeholder AACT snapshot generator
|  |- preprocess.py            stage 2 adapter (feature code or placeholder)
|  |- schema_stats.py          statistics + Great Expectations schema
|  |- validate.py              schema validation + drift checks + alerts
|  |- run_stage.py             runs one DVC stage (what each Airflow task does)
|  |- tracking.py              logging and per-step run records (+ MLflow)
|  |- alerts.py                SMTP email alerts
|  |- config.py                paths and params
|- schema/baseline/            frozen schema and statistics (committed)
|- tests/                      unit tests (pytest)
|- docker/Dockerfile           image the Composer tasks run
|- composer/                   Cloud Composer deployment (deploy.sh, README)
|- dvc.yaml, params.yaml       DVC stages and their parameters
|- dvc.lock                    hash of every output: the current data version
|- requirements.txt, .env.example
```

`data/`, `reports/`, `schema/generated/` and `logs/` are produced by the
pipeline and are not committed: DVC stores the data, git stores `dvc.lock`.

## Setup

Python 3.11 or later, from the repository root:

```bash
cd Data-Pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # optional: SMTP settings for email alerts
```

## Run the pipeline locally (no Airflow needed)

```bash
dvc repro            # runs every stage that changed
dvc repro validate   # or up to one stage
dvc dag              # shows the stage graph
dvc metrics show     # validation summary
```

`dvc repro` with nothing changed does nothing ("Data and pipelines are up to
date"). Change a parameter in `params.yaml` (for example `acquire.snapshot`)
and only the affected stages run again.

To use real AACT data instead of the placeholder snapshot, set
`acquire.source` to `zip` (with `acquire.zip` set to the export's path or URL)
or to `db` (with `AACT_USER` and `AACT_PASSWORD` in the environment). Both
need `src/ingestion/fetch_aact.py` in the checkout.

## 4. Pipeline orchestration (Airflow)

`dags/ctt_data_pipeline.py` defines the DAG `ctt_data_pipeline`:

- `acquire >> preprocess >> schema_stats >> validate >> publish`, run on the
  5th of every month at 06:00 UTC (after the monthly AACT export), or
  triggered by hand with a config such as
  `{"snapshot": "20261001", "source": "zip", "zip": "gs://.../export.zip"}`.
- Every task runs one DVC stage with `python -m scripts.run_stage <stage>`.
  On Cloud Composer each task is a `KubernetesPodOperator` that runs the
  pipeline image (`docker/Dockerfile`), so the pipeline's libraries never
  clash with Composer's own Airflow packages. `CTT_RUNNER=local` runs the
  same tasks as `BashOperator`s, for testing on a laptop.
- 2 retries with exponential backoff, a timeout per task and per run,
  `max_active_runs=1` (stages share `dvc.lock`), and an email to
  `CTT_ALERT_EMAILS` when a task fails.
- `validate` exits with an error on critical data problems, so the run stops
  before `publish` and no bad data version is recorded.

Because each Composer task runs in a fresh pod, `run_stage.py` keeps the
run's shared files in GCS (`CTT_STATE_URI`): it downloads `params.yaml` and
the latest `dvc.lock` and pulls the stage's inputs before the stage, then
pushes the outputs and uploads `dvc.lock`, logs and reports afterwards (logs
and reports also when the stage fails).

Deployment to Cloud Composer: see [composer/README.md](composer/README.md).

Test the DAG on a laptop:

```bash
pip install "apache-airflow==2.10.5" apache-airflow-providers-cncf-kubernetes \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt
export AIRFLOW_HOME=/tmp/airflow AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export CTT_RUNNER=local CTT_PYTHON=$PWD/.venv/bin/python
airflow db migrate
airflow dags test ctt_data_pipeline 2026-11-05 -c '{"snapshot": "20261101"}'
```

## 5. Data versioning with DVC

- DVC is initialised in this folder (`dvc init --subdir`), so it does not
  touch the rest of the repository.
- `dvc.yaml` declares each stage's code, input data, parameters and outputs.
  `dvc.lock` records the md5 hash of every output, so each commit of
  `dvc.lock` is one exact data version, and a model can be traced back to the
  snapshot it was trained on.
- The data itself goes to the DVC remote, a GCS bucket in `ctt-mlops`
  (`dvc push`); git only holds `dvc.lock` and the small `.gitignore` files.

### DVC remote

The remote URL in `.dvc/config` is a placeholder until the bucket name is
set. Set it once (everyone with access to the bucket can then pull):

```bash
dvc remote modify gcs url gs://<bucket>/dvc-store
gcloud auth application-default login   # Google account with access to ctt-mlops
dvc push      # upload the data versions in dvc.lock
dvc pull      # download them on another machine
```

Go back to an older data version: `git checkout <commit> -- dvc.lock && dvc
checkout`. On Cloud Composer every run also writes
`state/versions/<snapshot>/dvc.lock` to the bucket.

## 6. Tracking and logging

- **Logs:** every step logs to the console and to `logs/<step>.log`. On
  Composer the console output appears in the task log and in Cloud Logging.
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
- **Data versions:** `publish` writes `logs/versions/<snapshot>__<run_id>.json`
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

Freeze a baseline after a run on a snapshot you trust, and commit it:

```bash
python -m scripts.schema_stats --freeze --reason "first real AACT snapshot"
git add schema/baseline && git commit -m "Freeze baseline schema"
```

Until a baseline is frozen, `validate` checks the snapshot against its own
suite and says so in the report.

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

Both already match the code on the `nithish/s1-text-cleaning` branch: with
that branch merged, `preprocess` builds 151 features and `schema_stats`
generates 463 expectations, all passing on the sample snapshot.

## Tests

```bash
cd Data-Pipeline
pytest
```

40 tests cover the placeholder snapshot and features, statistics values, the
inferred schema, every critical and warning check (a dropped column,
duplicate or malformed ids, an invalid label, out-of-range values), drift
detection, the fail and alert policies, run records, email alerts (with a
fake SMTP server), parameter overrides, the stage graph and the DAG
structure for both runners (the DAG tests need Airflow installed and are
skipped otherwise).

## Reproducibility

Anyone with the repository can run the full pipeline offline on the
placeholder snapshot with `pip install -r requirements.txt && dvc repro`.
The placeholder snapshot is seeded, so the same parameters give the same
files and the same `dvc.lock` hashes. With access to the bucket, `dvc pull`
restores the exact data of any committed `dvc.lock`.
