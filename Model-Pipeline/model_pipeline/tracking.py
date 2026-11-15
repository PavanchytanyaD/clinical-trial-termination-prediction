"""Experiment tracking with MLflow.

Every pipeline run is one MLflow run (started by load_data, resumed by the
later stages); every hyperparameter trial is a nested run under it.

Where runs are logged (first match wins, see tracking_uri):
  1. MLFLOW_TRACKING_URI in the environment (CI, Vertex AI, a shared server)
  2. tracking.uri in params.yaml (the team's shared server or folder)
  3. mlruns/ in this run's work folder (copied to the bucket with the rest)

With 1 or 2, every run lands in the same place and can be compared in the
MLflow UI. With 3, runs stay separate; compare them afterwards with
`python -m model_pipeline.compare_runs`.
"""

import logging
import os
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)
_logged_uris = set()  # log the tracking address once per process

try:
    import mlflow
except ImportError:  # tests and quick runs without MLflow installed
    mlflow = None


def _is_file_store(uri):
    """A plain path or file: URI, which MLflow 3 only allows with a flag."""
    return "://" not in uri or uri.startswith("file:")


def _safe(uri):
    """The URI without a user name or password, for log lines."""
    parts = urlsplit(uri)
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        parts = parts._replace(netloc=host)
    return urlunsplit(parts)


def tracking_uri(ctx):
    """Where this run is logged: environment, then params.yaml, then the
    run's own work folder."""
    settings = ctx.params.get("tracking") or {}
    uri = os.environ.get("MLFLOW_TRACKING_URI") or settings.get("uri")
    if not uri:
        uri = (ctx.work / "mlruns").resolve().as_uri()
    if _is_file_store(uri):
        # MLflow 3 needs this flag to keep using a folder.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


def _setup(ctx):
    uri = tracking_uri(ctx)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(ctx.params["tracking"]["experiment"])
    if uri not in _logged_uris:
        logger.info("MLflow tracking at %s", _safe(uri))
        _logged_uris.add(uri)


@contextmanager
def pipeline_run(ctx):
    """The MLflow run of this pipeline run (created on first use)."""
    if mlflow is None:
        yield None
        return
    _setup(ctx)
    run_id = ctx.state.get("mlflow_run_id")
    with mlflow.start_run(run_id=run_id,
                          run_name=ctx.state.get("run_id")) as run:
        if run_id is None:
            ctx.update_state(mlflow_run_id=run.info.run_id)
        yield run


def log_params(ctx, params):
    with pipeline_run(ctx) as run:
        if run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})


def log_metrics(ctx, metrics, prefix=""):
    # MLflow names cannot contain "+" (as in logreg+group_thresholds)
    clean = {f"{prefix}{k}".replace("+", "__"): float(v)
             for k, v in metrics.items()
             if isinstance(v, (int, float)) and v == v}
    with pipeline_run(ctx) as run:
        if run:
            mlflow.log_metrics(clean)


def log_artifacts(ctx, folder, artifact_path=None):
    with pipeline_run(ctx) as run:
        if run:
            mlflow.log_artifacts(str(folder), artifact_path)


def set_tags(ctx, tags):
    with pipeline_run(ctx) as run:
        if run:
            mlflow.set_tags(tags)


def log_trial(ctx, family, number, hyperparams, metrics):
    """One hyperparameter trial as a nested run."""
    with pipeline_run(ctx) as run:
        if run is None:
            return
        with mlflow.start_run(run_name=f"{family}-{number:02d}",
                              nested=True):
            mlflow.set_tag("family", family)
            mlflow.log_params(hyperparams)
            mlflow.log_metrics({f"val_{k}": float(v)
                                for k, v in metrics.items()
                                if isinstance(v, (int, float)) and v == v})
