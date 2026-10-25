# Airflow for the data pipeline, the same on a laptop and on the GCP VM.
# Airflow 2.10.5 matches Cloud Composer 3. The pipeline's own libraries
# (pandas 3, Great Expectations, DVC) go into a separate virtualenv so they
# never clash with Airflow's pinned packages; the DAG's BashOperator tasks run
# that virtualenv's python (CTT_PYTHON).
#
# Build context: the repository root (see docker-compose.yaml).
FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    # DVC runs git in the mounted repository, owned by the host user
    && git config --system --add safe.directory '*'

USER airflow
COPY Data-Pipeline/requirements.txt /tmp/pipeline-requirements.txt
RUN python -m venv /opt/airflow/pipeline-venv \
    && /opt/airflow/pipeline-venv/bin/pip install --no-cache-dir \
       -r /tmp/pipeline-requirements.txt

# Pipeline python first on PATH: DVC stage commands call `python -m ...`.
# The airflow command itself keeps its own interpreter (absolute shebang).
ENV PATH="/opt/airflow/pipeline-venv/bin:${PATH}" \
    CTT_PYTHON=/opt/airflow/pipeline-venv/bin/python \
    CTT_RUNNER=local \
    CTT_PIPELINE_HOME=/opt/ctt/Data-Pipeline \
    PYTHONDONTWRITEBYTECODE=1 \
    GX_ANALYTICS_ENABLED=false
