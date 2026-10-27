#!/usr/bin/env bash
# Cloud Composer for the data pipeline. By default it works on the team's
# project ctt-mlops-project; set PROJECT (and optionally REGION, BUCKET) to
# deploy the same pipeline in your own Google Cloud project. Composer is
# billed every hour it exists and cannot be paused, so it is created for a
# run and deleted afterwards; the data, run state and Airflow's history (a
# snapshot) stay in the bucket.
# Read composer/README.md first; every step is safe to re-run.
#
#   [PROJECT=my-project] ./composer/deploy.sh <step>
#
# Once per project:
#   setup      enable the APIs, create the bucket and the service accounts
#              pipeline-sa (Composer, its pods, the Vertex AI job) and
#              cicd-sa (image builds), and give them their roles
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
#   schedule   show how often the pipeline runs, or change it, live:
#              SCHEDULE="@weekly" (or a cron such as "0 6 * * 1", "none" for
#              manual runs only, "default" for monthly)
#   ui         print the Airflow web UI address
#   lock       copy the latest run's dvc.lock into Data-Pipeline/dvc.lock, to
#              commit it: Git then keeps the history of the data versions
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
BUCKET="${BUCKET:-${PROJECT}-data}"     # DVC data and run state
STATE_URI="gs://${BUCKET}/data-pipeline/state"
DVC_REMOTE="gs://${BUCKET}/dvc-store"
SNAPSHOTS="gs://${BUCKET}/composer-snapshots"
ALERT_EMAILS="${ALERT_EMAILS:-}"
SECRET_NAME="ctt-pipeline-secrets"
ENV_SA="${ENV_SA:-pipeline-sa@${PROJECT}.iam.gserviceaccount.com}"
BUILD_SA="${BUILD_SA:-cicd-sa@${PROJECT}.iam.gserviceaccount.com}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # Data-Pipeline/
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

gc() { gcloud --project "${PROJECT}" "$@"; }
env_flags() { echo --environment "${ENV_NAME}" --location "${REGION}"; }

# Trigger a DAG; right after an upload the scheduler may not have registered
# it yet (DagNotFound), so retry for up to 5 minutes
trigger_dag() {
  local dag="$1" conf="$2" i
  for i in $(seq 1 10); do
    gc composer environments run "${ENV_NAME}" --location "${REGION}" \
      dags unpause -- "${dag}" >/dev/null 2>&1 || true
    if gc composer environments run "${ENV_NAME}" --location "${REGION}" \
         dags trigger -- "${dag}" -c "${conf}"; then
      return 0
    fi
    echo "${dag} not registered yet, retrying in 30 s (${i}/10)"
    sleep 30
  done
  return 1
}

# Build Data-Pipeline/docker/<dockerfile> with Cloud Build (repository root as
# context: src/ + Data-Pipeline/) and push it as :latest (and :$TAG).
# It builds the committed code (git archive HEAD). The archive goes to our own
# bucket and the build's first step fetches it, so CI needs no rights on Cloud
# Build's default source bucket (gcloud would try to create that bucket).
cloud_build() {
  gc artifacts repositories describe "${REPO}" --location "${REGION}" \
      >/dev/null 2>&1 \
    || gc artifacts repositories create "${REPO}" --location "${REGION}" \
         --repository-format docker
  # source archive and build config, both removed after the build
  local archive src
  archive="$(mktemp)"
  src="gs://${BUCKET}/cloudbuild-source/$(date -u +%Y%m%dT%H%M%S)-$$-$1.tgz"
  git -C "${REPO_ROOT}" archive --format=tar.gz -o "${archive}" HEAD
  gc storage cp "${archive}" "${src}" --quiet
  config="$(mktemp)"
  local tags="\"-t\", \"$2:latest\""
  local images="\"$2:latest\""
  if [ -n "${TAG}" ]; then
    tags="${tags}, \"-t\", \"$2:${TAG}\""
    images="${images}, \"$2:${TAG}\""
  fi
  cat > "${config}" <<EOF
steps:
  - name: gcr.io/cloud-builders/gsutil
    args: ["cp", "${src}", "/workspace/source.tgz"]
  - name: ubuntu
    args: ["bash", "-c", "tar -xzf source.tgz && rm source.tgz"]
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
  local rc=0
  gc builds submit --no-source --config "${config}" || rc=$?
  gc storage rm "${src}" --quiet || true
  rm -f "${config}" "${archive}"
  return "${rc}"
}

secret_data() {
  # SMTP_* / ALERT_* / AACT_* lines of .env with a value, as Kubernetes Secret
  # data (base64-encoded values)
  [ -f "${HERE}/.env" ] || return 0
  while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" == \#* || -z "${value}" ]] && continue
    echo "  ${key}: $(printf '%s' "${value}" | base64 | tr -d '\n')"
  done < <(grep -E '^(SMTP_|ALERT_|AACT_)' "${HERE}/.env")
}

# Create a service account if it does not exist yet
service_account() {
  local email="$1" name="${1%%@*}"
  gc iam service-accounts describe "${email}" >/dev/null 2>&1 \
    || gc iam service-accounts create "${name}" --display-name "$2"
}

step() {
  case "$1" in
    setup)
      gc services enable composer.googleapis.com artifactregistry.googleapis.com \
        cloudbuild.googleapis.com aiplatform.googleapis.com \
        iam.googleapis.com storage.googleapis.com
      gc storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
        || gc storage buckets create "gs://${BUCKET}" --location "${REGION}" \
             --uniform-bucket-level-access
      service_account "${ENV_SA}" "Data pipeline (Composer, pods, Vertex AI)"
      service_account "${BUILD_SA}" "CI/CD image builds"
      local number
      number="$(gc projects describe "${PROJECT}" --format='value(projectNumber)')"
      # ENV_SA: Composer worker; its pods read and write the bucket, pull
      # images, start the Vertex AI job running as itself, and write logs
      for role in roles/composer.worker roles/storage.objectAdmin \
          roles/artifactregistry.reader roles/aiplatform.user \
          roles/iam.serviceAccountUser roles/logging.logWriter; do
        gc projects add-iam-policy-binding "${PROJECT}" --condition=None \
          --member "serviceAccount:${ENV_SA}" --role "${role}" --format=none
      done
      # Composer's own service agent must be allowed to act as ENV_SA
      # (granted on the project: service-account-level IAM is not open to us)
      gc projects add-iam-policy-binding "${PROJECT}" --condition=None \
        --member "serviceAccount:service-${number}@cloudcomposer-accounts.iam.gserviceaccount.com" \
        --role roles/composer.ServiceAgentV2Ext --format=none
      # CI (GitHub Actions, signed in as BUILD_SA through the team's Workload
      # Identity pool): submit builds, stage the source in our bucket, push
      # images, and upload the DAG when Composer is up
      for role in roles/cloudbuild.builds.editor roles/artifactregistry.writer \
          roles/logging.logWriter roles/serviceusage.serviceUsageConsumer \
          roles/storage.objectAdmin roles/composer.user \
          roles/iam.serviceAccountUser; do
        gc projects add-iam-policy-binding "${PROJECT}" --condition=None \
          --member "serviceAccount:${BUILD_SA}" --role "${role}" --format=none
      done
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
      [ -n "$(secret_data)" ] && smtp="${SECRET_NAME}"
      # "^|^" makes | the separator, so ALERT_EMAILS may contain commas.
      # The project, bucket and images go to every pod, so the same image
      # runs in any project.
      gc composer environments update "${ENV_NAME}" --location "${REGION}" \
        --update-env-variables "^|^CTT_IMAGE=${IMAGE}:latest|CTT_STATE_URI=${STATE_URI}|CTT_DVC_REMOTE=${DVC_REMOTE}|CTT_GCP_PROJECT=${PROJECT}|CTT_REGION=${REGION}|CTT_BUCKET=${BUCKET}|CTT_GPU_IMAGE=${GPU_IMAGE}:latest|CTT_SERVICE_ACCOUNT=${ENV_SA}|CTT_ALERT_EMAILS=${ALERT_EMAILS}|CTT_SMTP_SECRET=${smtp}|CTT_SOURCE=${SOURCE:-sample}"
      ;;
    secret)
      local data file
      data="$(secret_data)"
      if [ -z "${data}" ]; then
        echo "no SMTP_/ALERT_/AACT_ values in Data-Pipeline/.env: no secret"
        return 0
      fi
      # Kubernetes Secret manifest, private and deleted right after
      file="$(mktemp)"
      chmod 600 "${file}"
      printf 'apiVersion: v1\nkind: Secret\nmetadata:\n  name: %s\ntype: Opaque\ndata:\n%s\n' \
        "${SECRET_NAME}" "${data}" > "${file}"
      gc beta composer environments user-workloads-secrets create \
          $(env_flags) --secret-file-path "${file}" 2>/dev/null \
        || gc beta composer environments user-workloads-secrets update \
             $(env_flags) --secret-file-path "${file}" \
        || { rm -f "${file}"; return 1; }
      rm -f "${file}"
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
      # Composer creates DAGs paused; trigger_dag unpauses first
      trigger_dag ctt_data_pipeline "${conf}"
      ;;
    freeze)
      trigger_dag ctt_freeze_baseline \
        "{\"reason\": \"${REASON:-trusted snapshot}\"}"
      ;;
    schedule)
      local var=(gc composer environments run "${ENV_NAME}" --location
                 "${REGION}" variables --)
      case "${SCHEDULE:-}" in
        "") "${var[@]}" get ctt_schedule 2>/dev/null \
              || echo "default: monthly, the 5th at 06:00 UTC (0 6 5 * *)" ;;
        default) "${var[@]}" delete ctt_schedule ;;
        *) "${var[@]}" set ctt_schedule "${SCHEDULE}" ;;
      esac
      ;;
    lock)
      gc storage cp "${STATE_URI}/dvc.lock" "${HERE}/dvc.lock"
      echo "commit Data-Pipeline/dvc.lock to record this data version in Git"
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
