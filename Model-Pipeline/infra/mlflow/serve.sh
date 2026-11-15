#!/bin/sh
# Start the MLflow server with the backend chosen at deploy time.
#   BACKEND=postgres  Cloud SQL through the Cloud Run socket
#                     (CLOUDSQL_INSTANCE, DB_NAME, DB_USER, DB_PASSWORD)
#   BACKEND=sqlite    a SQLite file (SQLITE_PATH, on the GCS volume)
# MLFLOW_ARTIFACT_ROOT: where runs store artifacts (gs://...). Clients write
# there directly (--no-serve-artifacts), so large files do not go through
# Cloud Run's 32 MB request limit.
set -eu

case "${BACKEND:-postgres}" in
  postgres)
    : "${CLOUDSQL_INSTANCE:?}" "${DB_PASSWORD:?}"
    STORE="postgresql+psycopg2://${DB_USER:-mlflow}:${DB_PASSWORD}@/${DB_NAME:-mlflow}?host=/cloudsql/${CLOUDSQL_INSTANCE}"
    ;;
  sqlite)
    STORE="sqlite:///${SQLITE_PATH:-/mnt/mlflow/mlflow.db}"
    ;;
  *)
    echo "unknown BACKEND ${BACKEND}" >&2
    exit 1
    ;;
esac

# Cloud Run has already checked the caller's ID token; the Host check is
# MLflow's guard against DNS rebinding, so the service's own names are
# allowed (and localhost for `gcloud run services proxy`).
exec mlflow server \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  --backend-store-uri "${STORE}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:?}" \
  --no-serve-artifacts \
  --workers "${WORKERS:-2}" \
  --allowed-hosts "${ALLOWED_HOSTS:-ctt-mlflow-*.run.app,localhost,localhost:*,127.0.0.1:*}"
