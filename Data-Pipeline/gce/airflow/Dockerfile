# Airflow for the data pipeline on the Compute Engine VM (gce/vm.sh).
# Airflow 2.10.5 matches Cloud Composer 3. The pipeline's own libraries
# (pandas 3, Great Expectations, DVC) go into a separate virtualenv so they
# never clash with Airflow's pinned packages; gce/run_task.sh runs each task
# with that virtualenv's python.
#
# Build context: the repository root (see docker-compose.yaml).
FROM apache/airflow:2.10.5-python3.11

USER airflow
COPY Data-Pipeline/requirements.txt /tmp/pipeline-requirements.txt
RUN python -m venv /opt/airflow/pipeline-venv \
    && /opt/airflow/pipeline-venv/bin/pip install --no-cache-dir \
       -r /tmp/pipeline-requirements.txt

ENV CTT_PYTHON=/opt/airflow/pipeline-venv/bin/python \
    CTT_RUNNER=vm \
    CTT_REPO=/opt/ctt \
    PYTHONDONTWRITEBYTECODE=1 \
    GX_ANALYTICS_ENABLED=false
