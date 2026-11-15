#!/usr/bin/env bash
# The team's MLflow tracking server on Cloud Run (service ctt-mlflow).
# Not public: only accounts with roles/run.invoker get in (pipeline-sa for
# the Vertex AI stages, and the team members in MEMBERS). Artifacts go to
# gs://<bucket>/mlflow-artifacts; runs, metrics and parameters go to the
# backend store chosen with BACKEND:
#
#   BACKEND=postgres (default)  Cloud SQL for PostgreSQL, db-f1-micro.
#                               About 9 USD a month while running; `stop`
#                               pauses it (only the disk is billed).
#   BACKEND=sqlite              a SQLite file on a Cloud Storage volume,
#                               one instance, one worker. Almost free, but
#                               slower and meant for one writer at a time
#                               (see README.md).
#
# Every step is safe to re-run. Read README.md first.
#
#   [PROJECT=...] [BACKEND=sqlite] [MEMBERS=user:a@x.com,user:b@y.com] \
#     ./Model-Pipeline/infra/mlflow/deploy_mlflow.sh <step>
#
#   all       setup, image, db, deploy, invokers, then print the URL
#   setup     enable the APIs, create mlflow-sa and give it its roles
#   image     build the server image with Cloud Build -> Artifact Registry
#   db        (postgres) Cloud SQL instance, database, user and password
#             secret
#   deploy    deploy (or update) the Cloud Run service
#   invokers  allow pipeline-sa and MEMBERS to call the service
#   url       print the service URL (the MLFLOW_TRACKING_URI to use)
#   proxy     open the UI on http://localhost:8080 with your gcloud login
#   stop      (postgres) pause the Cloud SQL instance (billing stops)
#   start     (postgres) start it again
set -euo pipefail

PROJECT="${PROJECT:-ctt-mlops-project}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT}-data}"
REPO="${REPO:-ctt-images}"
SERVICE="${SERVICE:-ctt-mlflow}"
BACKEND="${BACKEND:-postgres}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/mlflow-server"
TAG="${TAG:-3.18.0}"
ARTIFACTS="gs://${BUCKET}/mlflow-artifacts"
SQL_INSTANCE="${SQL_INSTANCE:-ctt-mlflow-db}"
SQL_TIER="${SQL_TIER:-db-f1-micro}"
DB_NAME="mlflow"
DB_USER="mlflow"
DB_SECRET="ctt-mlflow-db-password"
SQLITE_DIR="mlflow-db"              # folder in the bucket for BACKEND=sqlite
MLFLOW_SA="mlflow-sa@${PROJECT}.iam.gserviceaccount.com"
PIPELINE_SA="pipeline-sa@${PROJECT}.iam.gserviceaccount.com"
BUILD_SA="${BUILD_SA:-cicd-sa@${PROJECT}.iam.gserviceaccount.com}"
MEMBERS="${MEMBERS:-}"              # extra invokers, comma separated
HERE="$(cd "$(dirname "$0")" && pwd)"

gc() { gcloud --project "${PROJECT}" "$@"; }

step() {
  case "$1" in
    setup)
      gc services enable run.googleapis.com sqladmin.googleapis.com \
        secretmanager.googleapis.com artifactregistry.googleapis.com \
        cloudbuild.googleapis.com
      gc iam service-accounts describe "${MLFLOW_SA}" >/dev/null 2>&1 \
        || gc iam service-accounts create "${MLFLOW_SA%%@*}" \
             --display-name "MLflow tracking server (Cloud Run)"
      for role in roles/cloudsql.client roles/logging.logWriter; do
        gc projects add-iam-policy-binding "${PROJECT}" --condition=None \
          --member "serviceAccount:${MLFLOW_SA}" --role "${role}" --format=none
      done
      # the server reads and lists artifacts (UI) and, for sqlite, writes
      # the database file
      gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
        --member "serviceAccount:${MLFLOW_SA}" \
        --role roles/storage.objectAdmin --format=none
      gc artifacts repositories describe "${REPO}" --location "${REGION}" \
          >/dev/null 2>&1 \
        || gc artifacts repositories create "${REPO}" --location "${REGION}" \
             --repository-format docker
      ;;
    image)
      local config
      config="$(mktemp)"
      cat > "${config}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "-t", "${IMAGE}:${TAG}", "-t", "${IMAGE}:latest", "."]
images: ["${IMAGE}:${TAG}", "${IMAGE}:latest"]
serviceAccount: projects/${PROJECT}/serviceAccounts/${BUILD_SA}
options:
  logging: CLOUD_LOGGING_ONLY
EOF
      local rc=0
      gc builds submit "${HERE}" --config "${config}" \
        --gcs-source-staging-dir "gs://${BUCKET}/cloudbuild-source" || rc=$?
      rm -f "${config}"
      return "${rc}"
      ;;
    db)
      if [ "${BACKEND}" != "postgres" ]; then
        echo "BACKEND=${BACKEND}: no Cloud SQL needed"
        return 0
      fi
      gc sql instances describe "${SQL_INSTANCE}" >/dev/null 2>&1 \
        || gc sql instances create "${SQL_INSTANCE}" \
             --database-version POSTGRES_16 --edition enterprise \
             --tier "${SQL_TIER}" --region "${REGION}" \
             --storage-type HDD --storage-size 10 --no-storage-auto-increase \
             --availability-type zonal
      gc sql databases describe "${DB_NAME}" --instance "${SQL_INSTANCE}" \
          >/dev/null 2>&1 \
        || gc sql databases create "${DB_NAME}" --instance "${SQL_INSTANCE}"
      # the password lives only in Secret Manager; Cloud Run reads it from
      # there
      if ! gc secrets describe "${DB_SECRET}" >/dev/null 2>&1; then
        openssl rand -hex 24 | tr -d '\n' \
          | gc secrets create "${DB_SECRET}" --replication-policy automatic \
              --data-file=-
      fi
      local password
      password="$(gc secrets versions access latest --secret "${DB_SECRET}")"
      if gc sql users list --instance "${SQL_INSTANCE}" --format 'value(name)' \
           | grep -qx "${DB_USER}"; then
        gc sql users set-password "${DB_USER}" --instance "${SQL_INSTANCE}" \
          --password "${password}"
      else
        gc sql users create "${DB_USER}" --instance "${SQL_INSTANCE}" \
          --password "${password}"
      fi
      gc secrets add-iam-policy-binding "${DB_SECRET}" \
        --member "serviceAccount:${MLFLOW_SA}" \
        --role roles/secretmanager.secretAccessor --format=none
      ;;
    deploy)
      local args=(--image "${IMAGE}:${TAG}" --region "${REGION}"
                  --service-account "${MLFLOW_SA}" --no-allow-unauthenticated
                  --port 8080 --cpu 1 --memory 4Gi --min-instances 0
                  --timeout 300)
      if [ "${BACKEND}" = "postgres" ]; then
        local conn
        conn="$(gc sql instances describe "${SQL_INSTANCE}" \
                  --format 'value(connectionName)')"
        args+=(--add-cloudsql-instances "${conn}"
               --set-secrets "DB_PASSWORD=${DB_SECRET}:latest"
               --set-env-vars "^|^BACKEND=postgres|CLOUDSQL_INSTANCE=${conn}|DB_NAME=${DB_NAME}|DB_USER=${DB_USER}|MLFLOW_ARTIFACT_ROOT=${ARTIFACTS}|WORKERS=2"
               --max-instances 2)
      else
        # one instance and one worker: a single writer for the SQLite file
        args+=(--execution-environment gen2
               --add-volume "name=db,type=cloud-storage,bucket=${BUCKET},mount-options=only-dir=${SQLITE_DIR};uid=1000;gid=1000"
               --add-volume-mount "volume=db,mount-path=/mnt/mlflow"
               --set-env-vars "^|^BACKEND=sqlite|SQLITE_PATH=/mnt/mlflow/mlflow.db|MLFLOW_ARTIFACT_ROOT=${ARTIFACTS}|WORKERS=1"
               --max-instances 1 --concurrency 10)
      fi
      gc run deploy "${SERVICE}" "${args[@]}"
      ;;
    invokers)
      local m
      for m in "serviceAccount:${PIPELINE_SA}" ${MEMBERS//,/ }; do
        gc run services add-iam-policy-binding "${SERVICE}" --region "${REGION}" \
          --member "${m}" --role roles/run.invoker --format=none
        echo "invoker: ${m}"
      done
      ;;
    url)
      gc run services describe "${SERVICE}" --region "${REGION}" \
        --format 'value(status.url)'
      ;;
    proxy)
      gc run services proxy "${SERVICE}" --region "${REGION}" --port 8080
      ;;
    stop)
      gc sql instances patch "${SQL_INSTANCE}" --activation-policy NEVER --quiet
      ;;
    start)
      gc sql instances patch "${SQL_INSTANCE}" --activation-policy ALWAYS --quiet
      ;;
    all)
      step setup && step image && step db && step deploy && step invokers
      echo "MLFLOW_TRACKING_URI=$(step url)"
      ;;
    *)
      awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"
      exit 1
      ;;
  esac
}

step "${1:-}"
