#!/usr/bin/env bash
# One Airflow task on the VM (DAG runner "vm"), isolated like a Composer pod:
# the code is copied to a fresh folder, the stage runs there through
# scripts/run_stage.py (run state in GCS, data in the DVC remote), and the
# folder is deleted afterwards. Parallel tasks therefore never share
# params.yaml, dvc.lock or a DVC cache.
#
#   run_task.sh <stage> [--param section.key=value ...]
set -euo pipefail

REPO="${CTT_REPO:-/opt/ctt}"
PYTHON="${CTT_PYTHON:-python}"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

tar -C "${REPO}" -cf - \
    --exclude=.git --exclude=__pycache__ --exclude=.env \
    --exclude=Data-Pipeline/data --exclude=Data-Pipeline/logs \
    --exclude=Data-Pipeline/reports --exclude=Data-Pipeline/.dvc/cache \
    --exclude=Data-Pipeline/.dvc/tmp --exclude=Data-Pipeline/schema/generated \
    src configs Data-Pipeline \
  | tar -C "${work}" -xf -

cd "${work}/Data-Pipeline"
# no .git in the copy (as in the pipeline image): DVC runs without git
"${PYTHON}" -m dvc config --local core.no_scm true
# DVC stage commands call `python -m scripts.<stage>`
export PATH="$(dirname "${PYTHON}"):${PATH}"
exec "${PYTHON}" -m scripts.run_stage "$@"
