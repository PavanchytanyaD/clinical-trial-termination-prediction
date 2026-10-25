"""Run one DVC stage; this is what each Airflow task executes.

DVC defines the stages (dvc.yaml), tracks their inputs, outputs and
parameters (dvc.lock) and stores every data version in the DVC remote (a GCS
bucket). Airflow decides when and in what order stages run, retries them and
alerts on failure. Running a stage through DVC means an unchanged stage is
skipped and every output is versioned.

Two modes:

* local (PIPELINE_STATE_URI not set): runs ``dvc repro`` in this checkout.
* cloud (PIPELINE_STATE_URI=gs://bucket/prefix): used in Cloud Composer, where
  every task runs in a fresh pod. Before the stage, the run's params.yaml and
  the latest dvc.lock are downloaded from the state folder and the stage's
  inputs are pulled from the DVC remote; after it, the outputs are pushed and
  dvc.lock is uploaded again. Logs and reports are uploaded even when the
  stage fails, so a failed validation can be read.

    python -m scripts.run_stage acquire --param acquire.snapshot=20261001
    python -m scripts.run_stage preprocess
    python -m scripts.run_stage publish      # push data, write version record
"""

import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.config import (DVC_FILE, LOCK_FILE, LOG_DIR, PARAMS_FILE, ROOT,
                            VALIDATION_SUMMARY, load_params)
from scripts.tracking import run_id, setup_logging

logger = logging.getLogger("run_stage")

PLACEHOLDER_REMOTE = "CHANGE-ME"
UPLOAD_DIRS = ["logs", "reports"]


# ---------------------------------------------------------------- dvc.yaml
def stages():
    return yaml.safe_load(DVC_FILE.read_text())["stages"]


def _paths(entries):
    for e in entries or []:
        yield next(iter(e)) if isinstance(e, dict) else e


def stage_deps(stage):
    return list(_paths(stages()[stage].get("deps")))


def produced_by():
    """{output path: stage} for every DVC-cached output."""
    out = {}
    for name, s in stages().items():
        for key in ("outs", "metrics"):
            for e in s.get(key) or []:
                path = next(iter(e)) if isinstance(e, dict) else e
                cached = not (isinstance(e, dict) and
                              e[path] and e[path].get("cache") is False)
                if cached:
                    out[path] = name
    return out


# ---------------------------------------------------------------- params
def parse_override(text):
    key, sep, value = text.partition("=")
    section, dot, name = key.partition(".")
    if not sep or not dot or not section or not name or "." in name:
        raise ValueError(f"--param needs section.key=value, got {text!r}")
    return section, name, value


def _yaml_value(raw, current):
    """Keep the type the parameter already has: a string stays a string
    (snapshot "20261001" must not become a number)."""
    if isinstance(current, str) or current is None:
        return json.dumps(raw)
    value = yaml.safe_load(raw) if raw != "" else ""
    if isinstance(value, bool):
        return str(value).lower()
    return json.dumps(value) if isinstance(value, str) else str(value)


def set_params(overrides, path=PARAMS_FILE):
    """Write section.key=value overrides into params.yaml, editing only the
    changed lines so comments and layout stay as they are."""
    if not overrides:
        return
    path = Path(path)
    lines = path.read_text().splitlines()
    for text in overrides:
        section, name, raw = parse_override(text)
        current = (yaml.safe_load("\n".join(lines)) or {}).get(section, {})
        if not isinstance(current, dict):
            raise ValueError(f"params.yaml section {section!r} is not a map")
        value = _yaml_value(raw, current.get(name))
        start = next((i for i, line in enumerate(lines)
                      if line.rstrip() == f"{section}:"), None)
        if start is None:
            lines += ["", f"{section}:", f"  {name}: {value}"]
        else:
            end = next((i for i in range(start + 1, len(lines))
                        if lines[i] and not lines[i].startswith((" ", "#"))),
                       len(lines))
            pattern = re.compile(rf"^(\s+){re.escape(name)}:\s*([^#]*?)"
                                 rf"(\s+#.*)?$")
            for i in range(start + 1, end):
                m = pattern.match(lines[i])
                if m:
                    lines[i] = f"{m.group(1)}{name}: {value}{m.group(3) or ''}"
                    break
            else:
                lines.insert(end, f"  {name}: {value}")
        logger.info("param %s.%s = %s", section, name, value)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- dvc
def dvc(*args, check=True):
    cmd = [sys.executable, "-m", "dvc", *args]
    logger.info("$ dvc %s", " ".join(args))
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if check and rc != 0:
        raise RuntimeError(f"dvc {' '.join(args)} failed with exit code {rc}")
    return rc


def remote_url():
    """URL of the default DVC remote, or None."""
    def read(*args):
        return subprocess.run([sys.executable, "-m", "dvc", *args], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=60).stdout.strip()
    try:
        default = read("remote", "default")
        url = read("config", f"remote.{default}.url") if default else ""
    except (OSError, subprocess.SubprocessError):
        return None
    if url and PLACEHOLDER_REMOTE in url:
        logger.warning("DVC remote %s is still the placeholder %s; set the "
                       "real bucket with `dvc remote modify gcs url "
                       "gs://<bucket>/dvc-store`", default, url)
        return None
    return url


# ---------------------------------------------------------------- GCS state
def _gcs():
    from google.cloud import storage
    return storage.Client()


def _split(uri):
    if not uri.startswith("gs://"):
        raise ValueError(
            f"PIPELINE_STATE_URI must start with gs://, got {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.strip("/")


def download(uri, path):
    bucket, name = _split(uri)
    blob = _gcs().bucket(bucket).blob(name)
    if not blob.exists():
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(path))
    logger.info("downloaded %s", uri)
    return True


def upload(path, uri):
    bucket, name = _split(uri)
    _gcs().bucket(bucket).blob(name).upload_from_filename(str(path))
    logger.info("uploaded %s", uri)


def upload_tree(local_dir, uri):
    local_dir = Path(local_dir)
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            upload(p, f"{uri}/{p.relative_to(local_dir).as_posix()}")


class State:
    """The run's shared files in gs://.../state (cloud mode only)."""

    def __init__(self, uri, rid):
        self.uri = uri.rstrip("/")
        self.run = f"{self.uri}/runs/{rid}"

    def pull(self, have_overrides):
        download(f"{self.uri}/dvc.lock", LOCK_FILE)
        if not have_overrides:  # later tasks reuse the first task's params
            download(f"{self.run}/params.yaml", PARAMS_FILE)

    def push_params(self):
        upload(PARAMS_FILE, f"{self.run}/params.yaml")

    def push_lock(self):
        if LOCK_FILE.exists():
            upload(LOCK_FILE, f"{self.uri}/dvc.lock")
            upload(LOCK_FILE, f"{self.run}/dvc.lock")

    def push_artifacts(self, stage):
        for d in UPLOAD_DIRS:
            if (ROOT / d).exists():
                upload_tree(ROOT / d, f"{self.run}/{stage}/{d}")


# ---------------------------------------------------------------- stages
def run_stage(stage, overrides=()):
    if stage not in stages():
        raise ValueError(f"unknown stage {stage!r}; stages: {list(stages())}")
    state_uri = os.environ.get("PIPELINE_STATE_URI")
    state = State(state_uri, run_id()) if state_uri else None
    remote = remote_url()
    if state:
        if not remote:
            raise RuntimeError("cloud mode needs a real DVC remote; see "
                               "README 'DVC remote'")
        state.pull(bool(overrides))
    set_params(overrides)
    if state and overrides:
        state.push_params()

    if remote:
        upstream = produced_by()
        for dep in stage_deps(stage):
            if upstream.get(dep, stage) != stage:
                # missing data is reported by repro below
                dvc("pull", dep, check=False)
    rc = 1
    try:
        # --pull: an unchanged stage restores its outputs from the remote
        # instead of running again
        rc = dvc("repro", "--single-item", *(["--pull"] if remote else []),
                 stage, check=False)
        if rc == 0 and state:
            dvc("push", stage)
            state.push_lock()
    finally:
        if state:
            state.push_artifacts(stage)
    if rc != 0:
        raise RuntimeError(f"stage {stage} failed (dvc repro exit {rc}); "
                           f"see {LOG_DIR / (stage + '.log')}")


def version_record():
    """What was produced in this run: snapshot, output hashes, validation."""
    lock = yaml.safe_load(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    outs = {}
    for name, s in (lock.get("stages") or {}).items():
        for o in s.get("outs") or []:
            outs[o["path"]] = {k: o.get(k) for k in ("md5", "size", "nfiles")
                               if o.get(k) is not None}
    summary = (json.loads(VALIDATION_SUMMARY.read_text())
               if VALIDATION_SUMMARY.exists() else None)
    return {
        "run_id": run_id(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "snapshot": str(load_params().get("acquire", {}).get("snapshot")),
        "outputs": outs,
        "validation": summary,
    }


def publish():
    """Push every output to the DVC remote and record the data version."""
    state_uri = os.environ.get("PIPELINE_STATE_URI")
    state = State(state_uri, run_id()) if state_uri else None
    if state:
        state.pull(False)
    remote = remote_url()
    if remote:
        if state:
            for path in produced_by():
                dvc("pull", path, check=False)
        dvc("push")
    else:
        logger.warning("no DVC remote configured: data stays in the local "
                       "DVC cache only")
    record = version_record()
    versions = LOG_DIR / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    path = versions / f"{record['snapshot']}__{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2))
    logger.info("data version %s: %d outputs, validation %s", path.name,
                len(record["outputs"]),
                (record["validation"] or {}).get("success"))
    if state:
        base = f"{state.uri}/versions/{record['snapshot']}"
        upload(path, f"{base}/{record['run_id']}.json")
        upload(LOCK_FILE, f"{base}/dvc.lock")
        upload(LOCK_FILE, f"{base}/latest.dvc.lock")
    else:
        logger.info("commit the new data version: git add Data-Pipeline/"
                    "dvc.lock && git commit -m 'Data snapshot %s'",
                    record["snapshot"])
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", help="a stage in dvc.yaml, or publish")
    parser.add_argument("--param", action="append", default=[],
                        help="override params.yaml, e.g. "
                             "acquire.snapshot=20261001 (repeatable)")
    args = parser.parse_args()
    setup_logging(f"run_stage_{args.stage}")
    try:
        if args.stage == "publish":
            publish()
        else:
            run_stage(args.stage, args.param)
    except Exception:
        logger.exception("run_stage %s failed", args.stage)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
