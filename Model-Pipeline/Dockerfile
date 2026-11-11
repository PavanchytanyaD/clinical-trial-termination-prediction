# Model pipeline image: every Vertex AI pipeline stage runs this image.
# Build from the repository root (the code reuses src/ and the data
# pipeline's scripts/):
#   docker build -f Model-Pipeline/Dockerfile -t model-pipeline .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY Model-Pipeline/requirements.txt /app/Model-Pipeline/requirements.txt
RUN pip install -r requirements.txt -r Model-Pipeline/requirements.txt

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY Data-Pipeline/scripts/ /app/Data-Pipeline/scripts/
COPY Data-Pipeline/params.yaml /app/Data-Pipeline/params.yaml
COPY Model-Pipeline/ /app/Model-Pipeline/

WORKDIR /app/Model-Pipeline
CMD ["python", "-m", "model_pipeline.run", "--help"]
