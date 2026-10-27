#!/usr/bin/env bash
# The data pipeline on this machine: Airflow in Docker, no cloud account.
# Needs Docker with Compose v2 (Docker Desktop, or Docker Engine on Linux)
# and about 4 GB of memory for Docker. Read local/README.md first.
#
#   ./local/pipeline.sh <step>          (from the Data-Pipeline folder)
#
#   up        build the image (first time about 10 minutes), start Airflow
#             at http://localhost:8080 (user admin, password admin;
#             CTT_UI_PORT=8081 for another port)
#   trigger   start one run and wait for it (SNAPSHOT=YYYYMMDD
#             SOURCE=sample|zip|db ZIP=... SAMPLE_TRIALS=...)
#   freeze    make the latest run's schema and statistics the validation
#             baseline (REASON="...")
#   status    the runs and the state of each task of the latest run
#   down      stop Airflow (output/ and Airflow's history are kept)
#   clean     stop Airflow and delete its history and local/output/
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"   # Data-Pipeline/local
export AIRFLOW_UID="${AIRFLOW_UID:-$(id -u)}"
UI="http://localhost:${CTT_UI_PORT:-8080}"
compose() { docker compose -f "${HERE}/docker-compose.yaml" "$@"; }
af() { compose exec -T scheduler airflow "$@"; }

# Trigger a DAG and wait until the run ends; prints each task's state and
# where the run's files are (\$3, with RUN for the run id)
run_dag() {
  local dag="$1" conf="$2" where="$3" run_id state
  af dags unpause "${dag}" >/dev/null
  run_id="local__$(date -u +%Y%m%dT%H%M%S)"
  af dags trigger "${dag}" -r "${run_id}" -c "${conf}" >/dev/null
  echo "started ${dag} run ${run_id}; follow it at ${UI}"
  while true; do
    sleep 20
    state="$(af dags list-runs -d "${dag}" -o plain 2>/dev/null \
               | awk -v r="${run_id}" '$2 == r {print $3}' || true)"
    case "${state}" in
      success|failed) break ;;
    esac
  done
  af tasks states-for-dag-run "${dag}" "${run_id}" -o table \
    | awk 'NR == 1 || /\|/' || true
  echo "${dag} ${run_id}: ${state}"
  echo "${where//RUN/${run_id}}"
  [ "${state}" = success ]
}

case "${1:-}" in
  up)
    mkdir -p "${HERE}/output"
    compose up -d --build --wait
    echo "Airflow is up: ${UI} (admin / admin)"
    ;;
  trigger)
    run_dag ctt_data_pipeline "{\"snapshot\": \"${SNAPSHOT:-$(date +%Y%m%d)}\", \"source\": \"${SOURCE:-sample}\", \"zip\": \"${ZIP:-}\", \"sample_trials\": \"${SAMPLE_TRIALS:-}\"}" \
      "reports and logs: ${HERE}/output/state/runs/RUN/"
    ;;
  freeze)
    run_dag ctt_freeze_baseline "{\"reason\": \"${REASON:-trusted snapshot}\"}" \
      "baseline: ${HERE}/output/state/baseline/"
    ;;
  status)
    af dags list-runs -d ctt_data_pipeline -o table
    ;;
  down)
    compose down
    ;;
  clean)
    compose down --volumes
    rm -rf "${HERE}/output"
    ;;
  *)
    awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"
    exit 1
    ;;
esac
