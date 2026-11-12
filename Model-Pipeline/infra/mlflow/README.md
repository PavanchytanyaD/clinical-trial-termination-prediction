# Shared MLflow tracking server

One MLflow 3 server for the whole team, on Cloud Run (service `ctt-mlflow`,
project `ctt-mlops-project`, region `us-central1`). Every pipeline run on
Vertex AI, and any run from a laptop, logs to it, so all experiments are in
one place.

| Part | Where |
|---|---|
| Server image | `us-central1-docker.pkg.dev/ctt-mlops-project/ctt-images/mlflow-server:3.18.0` (`Dockerfile`, `serve.sh`) |
| Runs, parameters, metrics | Cloud SQL for PostgreSQL `ctt-mlflow-db` (db-f1-micro), or SQLite on a Cloud Storage volume (see below) |
| Artifacts (reports, charts) | `gs://ctt-mlops-project-data/mlflow-artifacts` |
| Server identity | `mlflow-sa` (Cloud SQL client, bucket access, the DB password secret) |
| Who can use it | `roles/run.invoker` on the service: `pipeline-sa` and the team members |

The service is not public (`--no-allow-unauthenticated`). Every request
needs a Google ID token; `model_pipeline/mlflow_auth.py` gets one and sets
`MLFLOW_TRACKING_TOKEN` whenever `MLFLOW_TRACKING_URI` is a `*.run.app`
address (from the service account on Vertex AI, or from
`gcloud auth print-identity-token` on a laptop).

Clients write artifacts straight to the bucket (the server runs with
`--no-serve-artifacts`), so whoever logs a run also needs write access to
`gs://ctt-mlops-project-data/mlflow-artifacts` (pipeline-sa has it).

## Backend: Postgres or SQLite

`BACKEND=postgres` (default) is the safe choice: a real database, several
server workers, no risk of a corrupted file. db-f1-micro with a 10 GB HDD
costs about 9 USD a month while it runs; `deploy_mlflow.sh stop` pauses it
between working sessions (only the disk is billed, under 1 USD) and
`start` brings it back in about a minute.

`BACKEND=sqlite` is the cheaper option: the database is one file,
`gs://ctt-mlops-project-data/mlflow-db/mlflow.db`, on a Cloud Storage FUSE
volume, and the service runs with at most one instance and one worker, so
there is a single writer. It costs next to nothing (Cloud Run scales to
zero), but Cloud Storage FUSE has no file locking and rewrites the file on
every commit, so it gets slow as the database grows and two servers
writing at once could corrupt it. Use it while the team is small and the
pipeline is the main writer; move to Postgres when it gets slow.

## Set it up (once)

Needs an account with Owner (or the IAM, Cloud Run, Cloud SQL and Secret
Manager admin roles) on the project, and the data pipeline's
`./Data-Pipeline/composer/deploy.sh setup` done before (bucket,
pipeline-sa, cicd-sa, the `ctt-images` repository).

```bash
gcloud auth login
gcloud config set project ctt-mlops-project

# everything: APIs, mlflow-sa, image, Cloud SQL, Cloud Run, invokers
MEMBERS="user:first@northeastern.edu,user:second@northeastern.edu" \
  ./Model-Pipeline/infra/mlflow/deploy_mlflow.sh all
# or the cheap variant
BACKEND=sqlite MEMBERS="user:..." ./Model-Pipeline/infra/mlflow/deploy_mlflow.sh all
```

The last line printed is the tracking URI, for example
`https://ctt-mlflow-431597074609.us-central1.run.app`. Then:

1. GitHub, Settings > Secrets and variables > Actions > Variables: add
   `MLFLOW_TRACKING_URI` with that address. CI passes it to every Vertex AI
   stage (`vertex/submit.py --mlflow-uri`).
2. On a laptop: `export MLFLOW_TRACKING_URI=<that address>` before
   `python -m model_pipeline.run ...` or `python vertex/submit.py`.

Single steps (`setup`, `image`, `db`, `deploy`, `invokers`) can be re-run
on their own; each one checks what already exists. Change the server after
editing `Dockerfile` or `serve.sh` with `deploy_mlflow.sh image` and
`deploy_mlflow.sh deploy`.

## Use it

```bash
# the UI on http://localhost:8080, signed in with your gcloud account
./Model-Pipeline/infra/mlflow/deploy_mlflow.sh proxy

# a quick check from the shell
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  "$(./Model-Pipeline/infra/mlflow/deploy_mlflow.sh url)/health"

# pause and resume the database (postgres only)
./Model-Pipeline/infra/mlflow/deploy_mlflow.sh stop
./Model-Pipeline/infra/mlflow/deploy_mlflow.sh start
```

Without `MLFLOW_TRACKING_URI` the pipeline still works: runs go to
`mlruns/` in the run's work folder, as before.

## Try the image locally

```bash
docker build -t ctt-mlflow Model-Pipeline/infra/mlflow
docker run --rm -p 8080:8080 -e BACKEND=sqlite -e SQLITE_PATH=/tmp/mlflow.db \
  -e MLFLOW_ARTIFACT_ROOT=/tmp/artifacts ctt-mlflow
```
