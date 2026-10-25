#!/usr/bin/env bash
# Run the data pipeline's Airflow on one Compute Engine VM: Docker Compose
# Airflow (gce/airflow/), each task isolated by gce/run_task.sh.
#
#   ./gce/vm.sh create     create the VM; it installs Docker, clones the repo
#                          and starts Airflow by itself (about 10 minutes)
#   ./gce/vm.sh status     VM state and Airflow containers
#   ./gce/vm.sh secrets    copy Data-Pipeline/.env (AACT, SMTP) to the VM
#   ./gce/vm.sh ui         open an SSH tunnel: Airflow at http://localhost:8081
#   ./gce/vm.sh password   print the Airflow admin password stored on the VM
#   ./gce/vm.sh trigger    start a run (SNAPSHOT=YYYYMMDD SOURCE=sample|zip|db)
#   ./gce/vm.sh freeze     make the latest run's schema and statistics the
#                          validation baseline, in GCS (REASON="...")
#   ./gce/vm.sh update     pull the latest code on the VM, refresh the boot
#                          script and restart Airflow
#   ./gce/vm.sh stop       stop the VM (compute billing stops, disk kept)
#   ./gce/vm.sh start      start it again (Airflow comes back by itself)
#   ./gce/vm.sh delete     delete the VM and its disk
#
# The VM runs as pipeline-sa (Storage, Vertex AI, Logging), so DVC reaches the
# bucket without keys. The web UI is bound to localhost on the VM; only SSH
# is reachable from outside.
set -euo pipefail

PROJECT="${PROJECT:-ctt-mlops-project}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-ctt-airflow}"
MACHINE="${MACHINE:-e2-standard-4}"     # 4 vCPU, 16 GB: full AACT fits
SA="${SA:-pipeline-sa@${PROJECT}.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/PavanchytanyaD/clinical-trial-termination-prediction.git}"
BRANCH="${BRANCH:-pavan/integration}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # Data-Pipeline/
APP=/opt/ctt-repo                          # repository on the VM

gc() { gcloud --project "${PROJECT}" "$@"; }
on_vm() { gc compute ssh "${VM}" --zone "${ZONE}" --quiet --command "$1"; }

startup_script() {
  cat <<EOF
#!/bin/bash
# Runs at every boot: install once, then (re)start Airflow.
set -e
if ! command -v docker >/dev/null; then
  apt-get update
  apt-get install -y ca-certificates curl git
  curl -fsSL https://get.docker.com | sh
fi
id ctt >/dev/null 2>&1 || useradd -m -s /bin/bash ctt
usermod -aG docker ctt
if [ ! -d ${APP}/.git ]; then
  git clone --branch ${BRANCH} ${REPO_URL} ${APP}
  chown -R ctt:ctt ${APP}
fi
cd ${APP}/Data-Pipeline/gce/airflow
if [ ! -f .env ]; then
  sed -e "s|^AIRFLOW_ADMIN_PASSWORD=.*|AIRFLOW_ADMIN_PASSWORD=\$(openssl rand -base64 12 | tr -d '/+=')|" \\
      -e "s|^AIRFLOW_UID=.*|AIRFLOW_UID=\$(id -u ctt)|" .env.example > .env
  chown ctt:ctt .env && chmod 600 .env
fi
sudo -u ctt docker compose up -d --build
EOF
}

case "${1:-}" in
  create)
    gc compute instances create "${VM}" --zone "${ZONE}" \
      --machine-type "${MACHINE}" \
      --image-family debian-12 --image-project debian-cloud \
      --boot-disk-size 100GB --boot-disk-type pd-balanced \
      --service-account "${SA}" --scopes cloud-platform \
      --labels app=ctt-data-pipeline \
      --metadata-from-file startup-script=<(startup_script)
    echo "Airflow comes up in about 10 minutes; check with: $0 status"
    ;;
  status)
    gc compute instances describe "${VM}" --zone "${ZONE}" \
      --format="value(status,machineType.basename())"
    on_vm "cd ${APP}/Data-Pipeline/gce/airflow && sudo -u ctt docker compose ps --format 'table {{.Service}}\t{{.Status}}'"
    ;;
  secrets)
    [ -f "${HERE}/.env" ] || { echo "no Data-Pipeline/.env to copy"; exit 1; }
    gc compute scp "${HERE}/.env" "${VM}:/tmp/pipeline.env" --zone "${ZONE}"
    on_vm "sudo install -o ctt -g ctt -m 600 /tmp/pipeline.env ${APP}/Data-Pipeline/.env && rm /tmp/pipeline.env && cd ${APP}/Data-Pipeline/gce/airflow && sudo -u ctt docker compose up -d"
    ;;
  ui)
    echo "Airflow UI: http://localhost:8081 (user admin; password: $0 password). Ctrl+C to close."
    gc compute ssh "${VM}" --zone "${ZONE}" --quiet -- -N -L 8081:localhost:8080
    ;;
  password)
    on_vm "sudo grep ^AIRFLOW_ADMIN_PASSWORD= ${APP}/Data-Pipeline/gce/airflow/.env | cut -d= -f2"
    ;;
  trigger)
    conf="{\"snapshot\": \"${SNAPSHOT:-$(date +%Y%m%d)}\", \"source\": \"${SOURCE:-sample}\", \"zip\": \"${ZIP:-}\"}"
    on_vm "cd ${APP}/Data-Pipeline/gce/airflow && sudo -u ctt docker compose exec -T scheduler airflow dags unpause ctt_data_pipeline && sudo -u ctt docker compose exec -T scheduler airflow dags trigger ctt_data_pipeline -c '${conf}'"
    ;;
  freeze)
    on_vm "cd ${APP}/Data-Pipeline/gce/airflow && sudo -u ctt docker compose exec -T scheduler /opt/ctt/Data-Pipeline/gce/run_task.sh freeze --reason '${REASON:-trusted snapshot}'"
    ;;
  update)
    gc compute instances add-metadata "${VM}" --zone "${ZONE}" \
      --metadata-from-file startup-script=<(startup_script)
    on_vm "cd ${APP} && sudo -u ctt git pull --ff-only && cd ${APP}/Data-Pipeline/gce/airflow && ([ -f .env ] || sudo cp ../../airflow/.env .env) && sudo -u ctt docker compose up -d --build --remove-orphans"
    ;;
  stop)   gc compute instances stop "${VM}" --zone "${ZONE}" ;;
  start)  gc compute instances start "${VM}" --zone "${ZONE}" ;;
  delete) gc compute instances delete "${VM}" --zone "${ZONE}" ;;
  *) awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 1 ;;
esac
