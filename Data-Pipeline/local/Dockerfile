# Airflow for running the whole pipeline on one machine (local/README.md).
# Airflow 2.10.5 matches Cloud Composer 3. The pipeline's own libraries
# (pandas 3, Great Expectations, DVC, and CPU-only torch for BioBERT) go into
# a separate virtualenv so they never clash with Airflow's pinned packages;
# each task runs that virtualenv's python (local/run_task.sh).
#
# Build context: the Data-Pipeline folder (docker-compose.yaml does this).
FROM apache/airflow:2.10.5-python3.11

USER airflow
COPY requirements.txt /tmp/pipeline-requirements.txt
RUN python -m venv /opt/airflow/pipeline-venv \
    && /opt/airflow/pipeline-venv/bin/pip install --no-cache-dir \
       torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/airflow/pipeline-venv/bin/pip install --no-cache-dir \
       -r /tmp/pipeline-requirements.txt transformers==5.5.0

# BioBERT weights in the image, so a run does not depend on Hugging Face
ENV HF_HOME=/opt/airflow/hf
RUN /opt/airflow/pipeline-venv/bin/python -c "from transformers import \
AutoModel, AutoTokenizer; m = 'dmis-lab/biobert-base-cased-v1.2'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

ENV HF_HUB_OFFLINE=1 \
    CTT_PYTHON=/opt/airflow/pipeline-venv/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    GX_ANALYTICS_ENABLED=false
