"""Compare pipeline runs logged to MLflow (guideline section 4).

Reads every pipeline run (not the nested hyperparameter trials) from one or
more MLflow stores and writes one table and one chart:

  run_comparison.csv  one row per run: when, status, selected model,
                      registry version, every logged metric and parameter
  run_comparison.png  the chosen metric (default test_pr_auc) per run

Works with a shared server or with the separate mlruns/ folders that runs
leave in their work folders (download them from the bucket first):

  python -m model_pipeline.compare_runs --uri http://<host>:5000
  python -m model_pipeline.compare_runs --uri runs/a/mlruns --uri runs/b/mlruns
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "ctt-model-development"
DEFAULT_METRIC = "test_pr_auc"
TAGS = ("selected_model", "registry_version")
PARENT_TAG = "mlflow.parentRunId"


def _to_uri(location):
    """Turn a folder path into a file: URI; leave server URLs alone."""
    if location.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        return location
    if "://" in location:
        return location
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    path = Path(location)
    if not path.exists():
        raise FileNotFoundError(f"MLflow folder not found: {path}")
    return path.resolve().as_uri()


def collect_runs(locations, experiment=DEFAULT_EXPERIMENT):
    """One row per pipeline run across all locations, newest first."""
    from mlflow.tracking import MlflowClient

    rows = []
    for location in locations:
        uri = _to_uri(location)
        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            logger.warning("no experiment '%s' in %s", experiment, location)
            continue
        for run in client.search_runs([exp.experiment_id], max_results=5000):
            if PARENT_TAG in run.data.tags:  # a hyperparameter trial
                continue
            row = {
                "source": location,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
                "status": run.info.status,
            }
            row.update({t: run.data.tags.get(t) for t in TAGS})
            row.update(run.data.metrics)
            row.update({f"param.{k}": v for k, v in run.data.params.items()})
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("start_time", ascending=False) \
        .reset_index(drop=True)


def pick_metric(runs, metric=DEFAULT_METRIC):
    """The metric to chart: the one asked for, else the first *pr_auc one."""
    if metric in runs.columns:
        return metric
    fallback = sorted(c for c in runs.columns if c.endswith("pr_auc"))
    if not fallback:
        raise KeyError(f"metric '{metric}' not logged by any run")
    logger.warning("metric '%s' not found, using '%s'", metric, fallback[0])
    return fallback[0]


def write_report(runs, out_dir, metric=DEFAULT_METRIC):
    """Write the table and chart; return the metric that was charted."""
    from model_pipeline import plots

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "run_comparison.csv", index=False)
    metric = pick_metric(runs, metric)
    scored = runs.dropna(subset=[metric])
    labels = scored["run_name"].fillna(scored["run_id"].str[:8])
    if "selected_model" in scored:
        labels = labels + " (" + scored["selected_model"].fillna("?") + ")"
    plots.bars(pd.Series(scored[metric].to_numpy(), index=labels),
               out_dir / "run_comparison.png",
               f"{metric} per pipeline run", metric)
    return metric


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uri", action="append", required=True,
                        help="MLflow server URL or mlruns folder; repeat "
                             "to combine several")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--out", default="reports/run_comparison")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        runs = collect_runs(args.uri, args.experiment)
        if runs.empty:
            raise RuntimeError("no pipeline runs found")
        metric = write_report(runs, args.out, args.metric)
        logger.info("compared %d runs on %s; wrote %s",
                    len(runs), metric, args.out)
    except Exception:
        logger.exception("run comparison failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
