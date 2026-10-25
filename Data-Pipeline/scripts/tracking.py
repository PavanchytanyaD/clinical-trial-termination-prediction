"""Logging and run tracking for every pipeline step.

``tracked_step(name)`` wraps a step's work:

* logs to the console and to logs/<step>.log (on Cloud Composer the console
  output goes to Cloud Logging),
* appends one JSON record per step to logs/runs/<run_id>.jsonl with the start
  and end time, duration, status, metrics and output fingerprints, so a run
  can be traced step by step,
* logs the metrics to MLflow too when MLFLOW_TRACKING_URI is set and mlflow
  is installed,
* on failure logs the traceback, emails an alert (unless an orchestrator
  that sends its own failure email is running the step) and re-raises.

The run id comes from PIPELINE_RUN_ID (the Airflow DAG passes its run_id);
outside Airflow it is the start time.
"""

import datetime as dt
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from scripts.alerts import send_alert
from scripts.config import LOG_DIR, REPO_ROOT, RUNS_DIR

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def setup_logging(name, log_dir=LOG_DIR, level=logging.INFO):
    """Log to the console and to <log_dir>/<name>.log."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(),
                logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")]
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers,
                        force=True)
    logging.getLogger("great_expectations").setLevel(logging.WARNING)


def run_id():
    rid = os.environ.get("PIPELINE_RUN_ID")
    if not rid:
        rid = dt.datetime.now().strftime("manual__%Y%m%dT%H%M%S")
        # later steps of the same dvc repro share it
        os.environ["PIPELINE_RUN_ID"] = rid
    return rid


def file_md5(path, chunk_bytes=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    """md5 and size of a file, or of every file in a folder."""
    path = Path(path)
    if path.is_file():
        return {"md5": file_md5(path), "bytes": path.stat().st_size}
    files = sorted(p for p in path.rglob("*") if p.is_file())
    h = hashlib.md5()
    for p in files:
        h.update(f"{p.relative_to(path)}:{file_md5(p)}".encode())
    return {"md5": h.hexdigest(),
            "bytes": sum(p.stat().st_size for p in files),
            "files": len(files)}


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _log_to_mlflow(record):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        logger.warning("MLFLOW_TRACKING_URI is set but mlflow is not "
                       "installed; skipping MLflow logging")
        return
    try:
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT",
                                             "ctt-data-pipeline"))
        with mlflow.start_run(run_name=f"{record['run_id']}:{record['step']}"):
            mlflow.set_tags({"step": record["step"],
                             "run_id": record["run_id"],
                             "status": record["status"]})
            numeric = {k: v for k, v in record["metrics"].items()
                       if isinstance(v, (int, float))
                       and not isinstance(v, bool)}
            if numeric:
                mlflow.log_metrics(numeric)
            mlflow.log_metric("duration_seconds", record["duration_seconds"])
    except Exception as e:  # tracking must never break the pipeline
        logger.warning("MLflow logging failed: %s", e)


@contextmanager
def tracked_step(name):
    """Set up logging and record the step's run; yields a dict for metrics."""
    setup_logging(name)
    record = {
        "run_id": run_id(),
        "step": name,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "status": "running",
        "git_commit": git_commit(),
        "host": platform.node(),
        "metrics": {},
        "outputs": {},
    }
    log = logging.getLogger(name)
    log.info("starting step %s (run %s)", name, record["run_id"])
    t0 = time.time()
    try:
        yield record
        record["status"] = "success"
    except BaseException as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        log.exception("step %s failed", name)
        if not (os.environ.get("PIPELINE_ORCHESTRATED")
                or getattr(e, "alerted", False)):
            send_alert(f"step {name} failed",
                       f"Run: {record['run_id']}\nStep: {name}\n\n"
                       f"{traceback.format_exc()}")
        raise
    finally:
        record["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds")
        record["duration_seconds"] = round(time.time() - t0, 3)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUNS_DIR / f"{record['run_id']}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        _log_to_mlflow(record)
        log.info("step %s %s in %.1fs", name, record["status"],
                 record["duration_seconds"])
