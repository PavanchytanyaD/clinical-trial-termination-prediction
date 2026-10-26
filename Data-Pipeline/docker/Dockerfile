# Image the Composer DAG runs for every stage (KubernetesPodOperator).
# Build from the repository root, so the image also has src/ (the ingestion,
# feature and bias code the stages call) and configs/:
#
#   docker build -f Data-Pipeline/docker/Dockerfile -t data-pipeline .
#
# The container runs one stage: python -m scripts.run_stage <stage>.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GX_ANALYTICS_ENABLED=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ctt
COPY Data-Pipeline/requirements.txt Data-Pipeline/requirements.txt
RUN pip install -r Data-Pipeline/requirements.txt

COPY src/ src/
# shared settings the stages read (bias_config.yaml for the bias stage)
COPY configs/ configs/
COPY Data-Pipeline/ Data-Pipeline/

WORKDIR /opt/ctt/Data-Pipeline
# No .git in the image: DVC runs without git. The remote and the run state
# live in GCS (scripts/run_stage.py downloads dvc.lock before each stage).
RUN dvc config --local core.no_scm true \
    && rm -rf data reports logs schema/generated .dvc/cache .dvc/tmp

ENTRYPOINT []
CMD ["python", "-m", "scripts.run_stage", "--help"]
