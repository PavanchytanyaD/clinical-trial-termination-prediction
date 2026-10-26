#!/usr/bin/env bash
# Cloud Composer for the data pipeline in the ctt-mlops project (project ID
# ctt-mlops-project). Composer is billed every hour it exists and cannot be
# paused, so it is created for a run and deleted afterwards; the data, run
# state and Airflow's history (a snapshot) stay in the bucket.
# Read composer/README.md first; every step is safe to re-run.
#
#   ./composer/deploy.sh <step>
#
# Once per project:
#   setup      enable the APIs, create the bucket, give pipeline-sa (the
#              environment's identity) the Composer roles
#
# Images (CI builds them on every push; by hand only when needed):
#   image      pipeline image (Composer pods) -> Artifact Registry
#   gpu-image  BioBERT GPU image (Vertex AI embeddings job) -> Artifact Registry
#
# For each working session:
#   up         create the environment (about 25 minutes), restore the last
#              snapshot, set the DAG's settings and secret, upload the DAGs
#   trigger    start one run (SNAPSHOT=YYYYMMDD SOURCE=sample|zip|db ZIP=...
#              SAMPLE_TRIALS=...)
#   freeze     make the latest run's schema and statistics the validation
#              baseline (REASON="...")
#   ui         print the Airflow web UI address
#   down       save a snapshot, then delete the environment (billing stops)
#
# Single steps of `up`: env, config, secret, dag
set -euo pipefail

PROJECT="${PROJECT:-ctt-mlops-project}"
REGION="${REGION:-us-central1}"
ENV_NAME="${ENV_NAME:-ctt-composer}"
REPO="${REPO:-ctt-images}"                # Artifact Registry repository
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/data-pipeline"
GPU_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/biobert-gpu"
TAG="${TAG:-}"                            # extra image tag, e.g. the git commit
BUCKET="${BUCKET:-ctt-mlops-project-data}"  # DVC data and run state
STATE_URI="gs://${BUCKET}/data-pipeline/state"
SNAPSHOTS="gs://${BUCKET}/composer-snapshots"
ALERT_EMAILS="${ALERT_EMAILS:-}"
SECRET_NAME="ctt-pipeline-secrets"
ENV_SA="${ENV_SA:-pipeline-sa@${PROJECT}.iam.gserviceaccount.com}"
BUILD_SA="${BUILD_SA:-cicd-sa@${PROJECT}.iam.gserviceaccount.com}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # Data-Pipeline/
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

gc() { gcloud --project "${PROJECT}" "$@"; }
env_flags() { echo --environment "${ENV_NAME}" --location "${REGION}"; }

# Build Data-Pipeline/docker/<dockerfile> with Cloud Build (repository root as
# context: src/ + Data-Pipeline/) and push it as :latest (and :$TAG).
cloud_build() {
  gc artifacts repositories describe "${REPO}" --location "${REGION}" \
      >/dev/null 2>&1 \
    || gc artifacts repositories create "${REPO}" --location "${REGION}" \
         --repository-format docker
  # build config, removed after the build
  config="$(mktemp)"
  local tags="\"-t\", \"$2:latest\""
  local images="\"$2:latest\""
  if [ -n "${TAG}" ]; then
    tags="${tags}, \"-t\", \"$2:${TAG}\""
    images="${images}, \"$2:${TAG}\""
  fi
  cat > "${config}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    env: ["DOCKER_BUILDKIT=1"]
    args: ["build", "-f", "Data-Pipeline/docker/$1", ${tags}, "."]
images: [${images}]
timeout: 3600s
# builds run as cicd-sa (Artifact Registry writer), not the Compute default
serviceAccount: projects/${PROJECT}/serviceAccounts/${BUILD_SA}
options:
  machineType: ${3:-UNSPECIFIED}
  logging: CLOUD_LOGGING_ONLY
EOF
  gc builds submit "${REPO_ROOT}" --config "${config}"
  rm -f "${config}"
}

secret_args() {
  # SMTP_* / ALERT_* / AACT_* lines of .env with a value, base64-encoded as
  # the Composer API expects
  [ -f "${HERE}/.env" ] || return 0
  while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" == \#* || -z "${value}" ]] && continue
    echo "--secret-data=${key}=$(printf '%s' "${value}" | base64 | tr -d '\n')"
  done < <(grep -E '^(SMTP_|ALERT_|AACT_)' "${HERE}/.env")
}

step() {
  case "$1" in
    setup)
      gc services enable composer.googleapis.com artifactregistry.googleapis.com \
        cloudbuild.googleapis.com aiplatform.googleapis.com
      gc storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
        || gc storage buckets create "gs://${BUCKET}" --location "${REGION}" \
             --uniform-bucket-level-access
      local number
      number="$(gc projects describe "${PROJECT}" --format='value(projectNumber)')"
      gc projects add-iam-policy-binding "${PROJECT}" --condition=None \
        --member "serviceAccount:${ENV_SA}" --role roles/composer.worker \
        --format=none
      # Composer's own service agent must be allowed to act as ENV_SA
      # (granted on the project: service-account-level IAM is not open to us)
      gc projects add-iam-policy-binding "${PROJECT}" --condition=None \
        --member "serviceAccount:service-${number}@cloudcomposer-accounts.iam.gserviceaccount.com" \
        --role roles/composer.ServiceAgentV2Ext --format=none
      ;;
    image)
      cloud_build Dockerfile "${IMAGE}"
      ;;
    gpu-image)
      # torch + CUDA + the BioBERT weights: a larger build machine is faster
      cloud_build Dockerfile.gpu "${GPU_IMAGE}" E2_HIGHCPU_8
      ;;
    env)
      # smallest Composer 3 environment, running as pipeline-sa
      gc composer environments describe "${ENV_NAME}" --location "${REGION}" \
          >/dev/null 2>&1 \
        || gc composer environments create "${ENV_NAME}" --location "${REGION}" \
             --image-version composer-3-airflow-2.10.5 \
             --environment-size small \
             --service-account "${ENV_SA}"
      local last
      last="$( (gc storage ls "${SNAPSHOTS}/" 2>/dev/null || true) | sort | tail -1)"
      if [ -n "${last}" ]; then
        echo "restoring Airflow's history from ${last}"
        gc composer environments snapshots load "${ENV_NAME}" \
          --location "${REGION}" --snapshot-path "${last%/}"
      fi
      ;;
    config)
      local smtp=""
      [ -n "$(secret_args)" ] && smtp="${SECRET_NAME}"
      gc composer environments update "${ENV_NAME}" --location "${REGION}" \
        --update-env-variables "CTT_IMAGE=${IMAGE}:latest,CTT_STATE_URI=${STATE_URI},CTT_ALERT_EMAILS=${ALERT_EMAILS},CTT_SMTP_SECRET=${smtp},CTT_SOURCE=${SOURCE:-sample}"
      ;;
    secret)
      local args=()
      while IFS= read -r a; do args+=("${a}"); done < <(secret_args)
      if [ "${#args[@]}" -eq 0 ]; then
        echo "no SMTP_/ALERT_/AACT_ values in Data-Pipeline/.env: no secret"
        return 0
      fi
      gc beta composer environments user-workloads-secrets create \
          $(env_flags) "${SECRET_NAME}" "${args[@]}" 2>/dev/null \
        || gc beta composer environments user-workloads-secrets update \
             $(env_flags) "${SECRET_NAME}" "${args[@]}"
      ;;
    dag)
      gc composer environments storage dags import $(env_flags) \
        --source "${HERE}/dags/ctt_data_pipeline.py"
      ;;
    up)
      step env && step secret && step config && step dag
      step ui
      ;;
    trigger)
      conf="{\"snapshot\": \"${SNAPSHOT:-$(date +%Y%m%d)}\", \"source\": \"${SOURCE:-sample}\", \"zip\": \"${ZIP:-}\", \"sample_trials\": \"${SAMPLE_TRIALS:-}\"}"
      gc composer environments run "${ENV_NAME}" --location "${REGION}" \
        dags trigger -- ctt_data_pipeline -c "${conf}"
      ;;
    freeze)
      gc composer environments run "${ENV_NAME}" --location "${REGION}" \
        dags trigger -- ctt_freeze_baseline \
        -c "{\"reason\": \"${REASON:-trusted snapshot}\"}"
      ;;
    ui)
      gc composer environments describe "${ENV_NAME}" --location "${REGION}" \
        --format='value(config.airflowUri)'
      ;;
    down)
      gc composer environments snapshots save "${ENV_NAME}" \
          --location "${REGION}" --snapshot-location "${SNAPSHOTS}" \
        || echo "snapshot failed; deleting anyway (data in GCS is not affected)"
      gc composer environments delete "${ENV_NAME}" --location "${REGION}" --quiet
      ;;
    *)
      awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"
      exit 1
      ;;
  esac
}

step "${1:-}"
