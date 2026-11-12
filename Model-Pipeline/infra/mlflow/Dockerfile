# MLflow tracking server for the team (Cloud Run service ctt-mlflow, see
# deploy_mlflow.sh and README.md). Build context: this folder.
#   docker build -t ctt-mlflow Model-Pipeline/infra/mlflow
# Local try-out with SQLite and a local artifact folder:
#   docker run -p 8080:8080 -e BACKEND=sqlite -e SQLITE_PATH=/tmp/mlflow.db \
#     -e MLFLOW_ARTIFACT_ROOT=/tmp/artifacts ctt-mlflow
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# same MLflow version as the pipeline's client (mlflow-skinny), the
# Postgres driver for Cloud SQL and the GCS client for the artifact store
RUN pip install mlflow==3.18.0 psycopg2-binary==2.9.13 \
    google-cloud-storage==3.17.1

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod 755 /usr/local/bin/serve.sh \
    && useradd --create-home --uid 1000 mlflow
USER mlflow
ENV PORT=8080
EXPOSE 8080
CMD ["/usr/local/bin/serve.sh"]
