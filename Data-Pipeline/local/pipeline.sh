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
#   schedule  show how often the pipeline runs, or change it, live:
#             schedule "@weekly" (or a cron such as "0 6 * * 1", "none" for
#             manual runs only, "default" for monthly)
#   status    the runs of the pipeline
#   down      stop Airflow (output/ and Airflow's history are kept)
#   clean     stop Airflow and delete its history and local/output/
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"   # Data-Pipeline/local
export AIRFLOW_UID="${AIRFLOW_UID:-$(id -u)}"
UI="http://localhost:${CTT_UI_PORT:-8080}"
compose() { docker compose -f "${HERE}/docker-compose.yaml" "$@"; }
af() { compose exec -T scheduler airflow "$@"; }

# Wait until the scheduler has parsed both DAGs (right after up, it has not)
wait_for_dags() {
  local i
  for i in $(seq 1 30); do
    if [ "$(af dags list -o plain 2>/dev/null \
            | grep -cE '^(ctt_data_pipeline|ctt_freeze_baseline) ')" = 2 ]; then
      return 0
    fi
    sleep 10
  done
  echo "the DAGs did not load; see: docker compose -f ${HERE}/docker-compose.yaml logs scheduler" >&2
  return 1
}

# Trigger a DAG and wait until the run ends; prints each task's state and
# where the run's files are (\$3, with RUN for the run id)
run_dag() {
  local dag="$1" conf="$2" where="$3" run_id state
  wait_for_dags
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
    wait_for_dags
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
  schedule)
    case "${2:-}" in
      "") af variables get ctt_schedule 2>/dev/null \
            || echo "default: monthly, the 5th at 06:00 UTC (0 6 5 * *)" ;;
      default) af variables delete ctt_schedule ;;
      *) af variables set ctt_schedule "$2" ;;
    esac
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
