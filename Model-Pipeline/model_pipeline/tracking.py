"""Experiment tracking with MLflow.

Every pipeline run is one MLflow run (started by load_data, resumed by the
later stages); every hyperparameter trial is a nested run under it. Set
MLFLOW_TRACKING_URI to use the team's MLflow server; otherwise runs are
written to mlruns/ in the work folder (copied to the bucket with the rest).
"""

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:  # tests and quick runs without MLflow installed
    mlflow = None


def _setup(ctx):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        # A folder of runs, copied to the bucket with the work folder.
        # MLflow 3 needs this flag to keep using a folder.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        uri = (ctx.work / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(ctx.params["tracking"]["experiment"])


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
