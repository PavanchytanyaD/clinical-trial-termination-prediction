"""Pipeline stage: BioBERT text embeddings for every trial.

Reads data/processed/text_all.jsonl (scripts/text.py) and writes
data/processed/embeddings/:

  ids.txt          one nct_id per line, the row order of every array
  <field>.npy      (n_trials, 768) float16 array per text field (title,
                   brief_summary, detailed_description, inclusion, exclusion,
                   outcomes); empty text gives a zero vector
  meta.json        model, fields, rows, windows per field, timing, job

The embedding code is src/features/biobert_embeddings.py (Task 1):
dmis-lab/biobert-base-cased-v1.2 used as is, long texts split into
overlapping 512-token windows and averaged. On Google Cloud it runs as a
Vertex AI custom job on a T4 GPU: this same module in worker mode, in the GPU
image (docker/Dockerfile.gpu). Input and output pass through the GCS bucket,
and the result comes back to the stage so DVC versions it like every other
output. params.yaml (section embeddings) sets the machine, the model
settings, and max_trials, which caps the number of trials for a quick test
(0 = all); an empty accelerator runs the job on CPU only.

Where the job runs comes from the environment, so the same code runs in any
Google Cloud project: PIPELINE_GCP_PROJECT, PIPELINE_REGION,
PIPELINE_BUCKET, PIPELINE_GPU_IMAGE, PIPELINE_SERVICE_ACCOUNT.
PIPELINE_EMBEDDINGS=local skips Vertex AI and runs BioBERT on this
machine's CPU (the local Docker Airflow does this).

    python -m scripts.embeddings                       # stage
    python -m scripts.embeddings --worker --input gs://... --output gs://...
        (what the Vertex AI job runs)
"""

import argparse
import datetime as dt
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

from scripts.config import PROCESSED_DIR, load_params
from scripts.tracking import fingerprint, tracked_step

logger = logging.getLogger("embeddings")

TEXT_JSONL = PROCESSED_DIR / "text_all.jsonl"
EMB_DIR = PROCESSED_DIR / "embeddings"
FIELDS = ["title", "brief_summary", "detailed_description", "inclusion",
          "exclusion", "outcomes"]
DEFAULTS = {
    "max_trials": 0, "batch_size": 32,
    "max_length": 512, "stride": 64,
    "model": "dmis-lab/biobert-base-cased-v1.2",
    "machine_type": "n1-standard-8", "accelerator": "NVIDIA_TESLA_T4",
    "timeout_hours": 8,
}


def deployment():
    """Where the Vertex AI job runs (environment, team project by default)."""
    env = os.environ.get
    project = env("PIPELINE_GCP_PROJECT") or "ctt-mlops-project"
    region = env("PIPELINE_REGION") or "us-central1"
    return {
        "backend": env("PIPELINE_EMBEDDINGS") or "vertex",
        "project": project, "region": region,
        "bucket": env("PIPELINE_BUCKET") or f"{project}-data",
        "image": env("PIPELINE_GPU_IMAGE") or
        f"{region}-docker.pkg.dev/{project}/ctt-images/biobert-gpu:latest",
        "service_account": env("PIPELINE_SERVICE_ACCOUNT") or
        f"pipeline-sa@{project}.iam.gserviceaccount.com",
    }


# settings the worker needs; the rest only says where the job runs
EMBED_KEYS = ["model", "batch_size", "max_length", "stride", "max_trials"]


# ---------------------------------------------------------------- embedding
def read_texts(path, max_trials=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_trials and len(rows) >= max_trials:
                break
            rows.append(json.loads(line))
    return rows


def embed_rows(rows, out_dir, params):
    """Embed every field of rows into out_dir (ids.txt, <field>.npy)."""
    from src.features.biobert_embeddings import embed_texts, load_biobert

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, device = load_biobert(params["model"])
    logger.info("BioBERT on %s for %d trials", device, len(rows))
    (out_dir / "ids.txt").write_text(
        "\n".join(r["nct_id"] for r in rows) + "\n")
    windows, seconds = {}, {}
    for field in FIELDS:
        t0 = time.time()
        texts = [r.get(field) or "" for r in rows]
        vecs, n_windows = embed_texts(
            texts, tokenizer, model, device, max_length=params["max_length"],
            stride=params["stride"], batch_size=params["batch_size"])
        np.save(out_dir / f"{field}.npy", vecs.astype(np.float16))
        windows[field] = int(n_windows.sum())
        seconds[field] = round(time.time() - t0, 1)
        logger.info("%s: %d windows in %.0f s", field, windows[field],
                    seconds[field])
    meta = {"model": params["model"], "fields": FIELDS, "rows": len(rows),
            "dim": int(model.config.hidden_size), "dtype": "float16",
            "device": str(device), "windows": windows, "seconds": seconds,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- GCS
def _gcs():
    from google.cloud import storage
    return storage.Client()


def _split(uri):
    bucket, _, name = uri[5:].partition("/")
    return bucket, name.rstrip("/")


def gcs_upload(path, uri):
    bucket, name = _split(uri)
    _gcs().bucket(bucket).blob(name).upload_from_filename(str(path))


def gcs_download(uri, path):
    bucket, name = _split(uri)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _gcs().bucket(bucket).blob(name).download_to_filename(str(path))


def gcs_upload_dir(folder, uri):
    for p in sorted(Path(folder).iterdir()):
        gcs_upload(p, f"{uri.rstrip('/')}/{p.name}")


def gcs_download_dir(uri, folder):
    bucket, prefix = _split(uri)
    for blob in _gcs().list_blobs(bucket, prefix=prefix + "/"):
        name = blob.name[len(prefix) + 1:]
        if name:
            Path(folder).mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(Path(folder) / name))


# ---------------------------------------------------------------- Vertex AI
def _session(project):
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=project)
    return AuthorizedSession(creds)


def vertex_job_spec(p, input_uri, output_uri, run_name):
    machine = {"machineType": p["machine_type"]}
    if p.get("accelerator"):
        machine.update(acceleratorType=p["accelerator"], acceleratorCount=1)
    return {
        "displayName": f"ctt-biobert-{run_name}"[:120],
        "labels": {"app": "ctt-data-pipeline", "stage": "embeddings"},
        "jobSpec": {
            "workerPoolSpecs": [{
                "machineSpec": machine,
                "replicaCount": "1",
                "diskSpec": {"bootDiskType": "pd-ssd",
                             "bootDiskSizeGb": 100},
                "containerSpec": {
                    "imageUri": p["image"],
                    "command": ["python", "-m", "scripts.embeddings"],
                    "args": ["--worker", "--input", input_uri,
                             "--output", output_uri, "--settings",
                             json.dumps({k: p[k] for k in EMBED_KEYS})],
                },
            }],
            "serviceAccount": p["service_account"],
            "scheduling": {"timeout": f"{int(p['timeout_hours']) * 3600}s"},
        },
    }


DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
               "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def _stop(signum, frame):
    raise SystemExit(f"stopped by signal {signum}")


def wait_for_job(session, job_url, poll_seconds=60):
    """Poll the job until it ends. If this process is stopped or fails
    (an Airflow timeout sends SIGTERM), cancel the job so the GPU is not
    left running."""
    previous = signal.signal(signal.SIGTERM, _stop)
    try:
        while True:
            state = session.get(job_url).json()
            logger.info("Vertex AI job state: %s", state.get("state"))
            if state.get("state") in DONE_STATES:
                return state
            time.sleep(poll_seconds)
    except BaseException:
        logger.warning("cancelling Vertex AI job %s", job_url)
        session.post(f"{job_url}:cancel")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_on_vertex(p, run_name, poll_seconds=60):
    """Upload the text, run the GPU job, download the embeddings."""
    base = f"gs://{p['bucket']}/data-pipeline/embeddings/{run_name}"
    input_uri, output_uri = f"{base}/text_all.jsonl", f"{base}/out"
    gcs_upload(TEXT_JSONL, input_uri)
    logger.info("uploaded text to %s", input_uri)
    api = (f"https://{p['region']}-aiplatform.googleapis.com/v1/projects/"
           f"{p['project']}/locations/{p['region']}/customJobs")
    session = _session(p["project"])
    spec = vertex_job_spec(p, input_uri, output_uri, run_name)
    r = session.post(api, json=spec)
    if r.status_code >= 300:
        raise RuntimeError(f"Vertex AI refused the job: {r.status_code} "
                           f"{r.text[:500]}")
    job = r.json()["name"]
    job_url = f"https://{p['region']}-aiplatform.googleapis.com/v1/{job}"
    logger.info("submitted Vertex AI job %s", job)
    state = wait_for_job(session, job_url, poll_seconds)
    if state["state"] != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Vertex AI job {job} ended {state['state']}: "
                           f"{state.get('error')}")
    if EMB_DIR.exists():
        shutil.rmtree(EMB_DIR)
    gcs_download_dir(output_uri, EMB_DIR)
    meta = json.loads((EMB_DIR / "meta.json").read_text())
    meta["vertex_job"] = job
    (EMB_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def run_local(p):
    """BioBERT on this machine (CPU, or a GPU if there is one)."""
    if EMB_DIR.exists():
        shutil.rmtree(EMB_DIR)
    return embed_rows(read_texts(TEXT_JSONL, p["max_trials"]), EMB_DIR, p)


# ---------------------------------------------------------------- entry points
def worker(input_uri, output_uri, settings):
    """Inside the Vertex AI job: GCS text in, GCS embeddings out."""
    p = {**DEFAULTS, **settings}
    with tempfile.TemporaryDirectory() as tmp:
        text_file = Path(tmp) / "text_all.jsonl"
        gcs_download(input_uri, text_file)
        meta = embed_rows(read_texts(text_file, p["max_trials"]),
                          Path(tmp) / "out", p)
        gcs_upload_dir(Path(tmp) / "out", output_uri)
    logger.info("uploaded %d embeddings to %s", meta["rows"], output_uri)


def stage():
    with tracked_step("embeddings") as run:
        p = {**DEFAULTS, **load_params().get("embeddings", {}),
             **deployment()}
        if p["backend"] == "local":
            meta = run_local(p)
        else:
            meta = run_on_vertex(
                p, re.sub(r"[^a-z0-9-]+", "-", run["run_id"].lower()))
        run["metrics"].update({"rows": meta["rows"],
                               **{f"seconds_{k}": v
                                  for k, v in meta["seconds"].items()}})
        run["outputs"]["data/processed/embeddings"] = fingerprint(EMB_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--settings", default="{}",
                        help="JSON: model, batch_size, max_length, stride, "
                             "max_trials")
    args = parser.parse_args()
    if args.worker:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: "
                                   "%(message)s")
        worker(args.input, args.output, json.loads(args.settings))
    else:
        stage()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("embeddings failed")
        raise SystemExit(1)
