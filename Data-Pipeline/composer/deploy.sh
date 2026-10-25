#!/usr/bin/env bash
# Deploy the data pipeline to Cloud Composer in the ctt-mlops project
# (project ID ctt-mlops-project).
# Read composer/README.md first; every step is safe to re-run.
#
#   ALERT_EMAILS=a@x.com,b@y.com ./composer/deploy.sh <step>
#
# Steps (run in this order the first time):
#   bucket     create the data bucket (DVC remote + run state), if missing
#   image      build the pipeline image with Cloud Build, push to Artifact Registry
#   gpu-image  build the BioBERT GPU image the embeddings stage runs on
#              Vertex AI (needed by the Docker/VM Airflow as well)
#   env        create the Composer 3 environment (BILLED while it exists)
#   config     set the DAG's environment variables on the environment
#   secret     create the SMTP/AACT Kubernetes secret from Data-Pipeline/.env
#   dag        upload dags/ctt_data_pipeline.py
#   trigger    start one run (SNAPSHOT=YYYYMMDD SOURCE=sample|zip|db ZIP=...)
#   delete     delete the Composer environment (stops the billing)
set -euo pipefail

PROJECT="${PROJECT:-ctt-mlops-project}"
REGION="${REGION:-us-central1}"
ENV_NAME="${ENV_NAME:-ctt-composer}"
REPO="${REPO:-ctt-images}"                # Artifact Registry repository
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/data-pipeline:latest"
GPU_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/biobert-gpu:latest"
BUCKET="${BUCKET:-ctt-mlops-project-data}"  # DVC data and run state
STATE_URI="gs://${BUCKET}/data-pipeline/state"
ALERT_EMAILS="${ALERT_EMAILS:-}"
SECRET_NAME="ctt-pipeline-secrets"
BUILD_SA="${BUILD_SA:-cicd-sa@${PROJECT}.iam.gserviceaccount.com}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # Data-Pipeline/
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

gc() { gcloud --project "${PROJECT}" "$@"; }

# Build Data-Pipeline/docker/<dockerfile> with Cloud Build (repository root as
# context: src/ + Data-Pipeline/) and push it to Artifact Registry.
cloud_build() {
  gc artifacts repositories describe "${REPO}" --location "${REGION}" \
      >/dev/null 2>&1 \
    || gc artifacts repositories create "${REPO}" --location "${REGION}" \
         --repository-format docker
  # build config, removed after the build
  config="$(mktemp)"
  cat > "${config}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    env: ["DOCKER_BUILDKIT=1"]
    args: ["build", "-f", "Data-Pipeline/docker/$1", "-t", "$2", "."]
images: ["$2"]
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

case "${1:-}" in
  bucket)
    gc storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
      || gc storage buckets create "gs://${BUCKET}" --location "${REGION}" \
           --uniform-bucket-level-access
    ;;
  image)
    cloud_build Dockerfile "${IMAGE}"
    ;;
  gpu-image)
    # torch + CUDA + the BioBERT weights: a larger build machine is faster
    cloud_build Dockerfile.gpu "${GPU_IMAGE}" E2_HIGHCPU_8
    ;;
  env)
    # Smallest Composer 3 environment; it is billed every hour it exists.
    gc composer environments create "${ENV_NAME}" --location "${REGION}" \
      --image-version composer-3-airflow-2.10.5 \
      --environment-size small \
      --env-variables "CTT_RUNNER=k8s"
    ;;
  config)
    gc composer environments update "${ENV_NAME}" --location "${REGION}" \
      --update-env-variables "CTT_RUNNER=k8s,CTT_IMAGE=${IMAGE},CTT_STATE_URI=${STATE_URI},CTT_ALERT_EMAILS=${ALERT_EMAILS},CTT_SMTP_SECRET=${SECRET_NAME},CTT_SOURCE=${SOURCE:-sample}"
    ;;
  secret)
    # Composer 3 runs the pods in its own cluster; the secret is created
    # through the Composer API, from the SMTP_* / ALERT_* / AACT_* lines of .env
    [ -f "${HERE}/.env" ] || { echo "create Data-Pipeline/.env from .env.example first"; exit 1; }
    args=()
    while IFS='=' read -r key value; do
      [[ -z "${key}" || "${key}" == \#* || -z "${value}" ]] && continue
      # the Composer API expects base64-encoded secret values
      args+=("--secret-data=${key}=$(printf '%s' "${value}" | base64 | tr -d '\n')")
    done < <(grep -E '^(SMTP_|ALERT_|AACT_)' "${HERE}/.env")
    gc beta composer environments user-workloads-secrets create \
      --environment "${ENV_NAME}" --location "${REGION}" "${SECRET_NAME}" \
      "${args[@]}" \
      || gc beta composer environments user-workloads-secrets update \
           --environment "${ENV_NAME}" --location "${REGION}" "${SECRET_NAME}" \
           "${args[@]}"
    ;;
  dag)
    gc composer environments storage dags import --environment "${ENV_NAME}" \
      --location "${REGION}" --source "${HERE}/dags/ctt_data_pipeline.py"
    ;;
  trigger)
    conf="{\"snapshot\": \"${SNAPSHOT:-$(date +%Y%m01)}\", \"source\": \"${SOURCE:-sample}\", \"zip\": \"${ZIP:-}\"}"
    gc composer environments run "${ENV_NAME}" --location "${REGION}" \
      dags trigger -- ctt_data_pipeline -c "${conf}"
    ;;
  delete)
    gc composer environments delete "${ENV_NAME}" --location "${REGION}"
    ;;
  *)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
