# Prediction server for the registered model (Vertex AI custom container).
# Build from the repository root:
#   docker build -f Model-Pipeline/serving/Dockerfile -t model-serving .
# Without torch (smaller, sklearn/XGBoost models only):
#   docker build -f Model-Pipeline/serving/Dockerfile \
#     --build-arg WITH_TORCH=false -t model-serving .
# Local run with a packaged model (the test stage's artifact/ folder):
#   docker run -p 8080:8080 -e MODEL_DIR=/model \
#     -v "$PWD/Model-Pipeline/runs/local/artifact:/model:ro" model-serving
FROM python:3.11-slim

ARG WITH_TORCH=true
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Model-Pipeline
COPY Model-Pipeline/serving/requirements.txt serving/requirements.txt
RUN if [ "${WITH_TORCH}" = "true" ]; then \
        pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && pip install -r serving/requirements.txt

# model_pipeline/ holds artifact.py and the model classes a pickled model
# may refer to (e.g. the PyTorch wrapper)
COPY Model-Pipeline/model_pipeline/ model_pipeline/
COPY Model-Pipeline/serving/ serving/
RUN useradd --create-home --uid 1000 app
USER app

ENV AIP_HTTP_PORT=8080
EXPOSE 8080
CMD ["python", "-m", "serving.app"]
