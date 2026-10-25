"""Run one DVC stage; this is what each Airflow task executes.

DVC defines the stages (dvc.yaml), tracks their inputs, outputs and
parameters (dvc.lock) and stores every data version in the DVC remote (a GCS
bucket). Airflow decides when and in what order stages run, retries them and
alerts on failure. Running a stage through DVC means an unchanged stage is
skipped and every output is versioned.

Every task starts from a fresh copy of the code (a Composer pod, or a
temporary folder on the VM: gce/run_task.sh), so the run's shared state lives
in GCS (PIPELINE_STATE_URI, default gs://ctt-mlops-project-data/data-pipeline/
state). Before the stage, the run's params.yaml and the latest dvc.lock are
downloaded from there and the stage's inputs are pulled from the DVC remote;
after it, the outputs are pushed and dvc.lock is merged back. Logs and
reports are uploaded even when the stage fails, so a failed validation can be
read.

    python -m scripts.run_stage acquire --param acquire.snapshot=20261001
    python -m scripts.run_stage preprocess
    python -m scripts.run_stage publish      # push data, write version record
    python -m scripts.run_stage freeze --reason "first real AACT snapshot"

freeze makes the latest run's schema and statistics the validation baseline:
it is stored in the state folder (state/baseline/), and every run downloads
it before its stages, so validate always compares against it.
"""

import argparse
import datetime as dt
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from scripts.config import (BASELINE_DIR, DVC_FILE, LOCK_FILE, LOG_DIR,
                            PARAMS_FILE, ROOT,
                            VALIDATION_SUMMARY, load_params)
from scripts.tracking import run_id, setup_logging

logger = logging.getLogger("run_stage")

PLACEHOLDER_REMOTE = "CHANGE-ME"
UPLOAD_DIRS = ["logs", "reports"]
DEFAULT_STATE_URI = "gs://ctt-mlops-project-data/data-pipeline/state"


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


def read_versioned(uri):
    """(text, generation) of a GCS object; (None, 0) if it does not exist."""
    from google.api_core.exceptions import NotFound
    bucket, name = _split(uri)
    blob = _gcs().bucket(bucket).blob(name)
    try:
        blob.reload()
        return blob.download_as_text(if_generation_match=blob.generation), \
            blob.generation
    except NotFound:
        return None, 0


def write_if_generation(uri, text, generation):
    """Write text only if the object is still at that generation (0 = must
    not exist yet). Returns False when another writer got there first."""
    from google.api_core.exceptions import PreconditionFailed
    bucket, name = _split(uri)
    try:
        _gcs().bucket(bucket).blob(name).upload_from_string(
            text, if_generation_match=generation)
    except PreconditionFailed:
        return False
    return True


def merge_lock_entry(shared_text, own_lock, stage):
    """The shared dvc.lock with this stage's entry replaced by ours."""
    shared = yaml.safe_load(shared_text or "") or {}
    shared.setdefault("schema", own_lock.get("schema", "2.0"))
    shared.setdefault("stages", {})[stage] = own_lock["stages"][stage]
    return yaml.safe_dump(shared, sort_keys=False)


def download_tree(uri, folder):
    """Download every object under uri/ into folder; False if none."""
    bucket, prefix = _split(uri)
    found = False
    for blob in _gcs().list_blobs(bucket, prefix=prefix + "/"):
        name = blob.name[len(prefix) + 1:]
        if name:
            target = Path(folder) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            found = True
    return found


def upload_tree(folder, uri):
    folder = Path(folder)
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            upload(p, f"{uri}/{p.relative_to(folder).as_posix()}")


class State:
    """The run's shared files in gs://.../state."""

    def __init__(self, uri, rid):
        self.uri = uri.rstrip("/")
        self.run = f"{self.uri}/runs/{rid}"

    def pull(self, have_overrides):
        download(f"{self.uri}/dvc.lock", LOCK_FILE)
        if not have_overrides:  # later tasks reuse the first task's params
            download(f"{self.run}/params.yaml", PARAMS_FILE)
        if download_tree(f"{self.uri}/baseline", BASELINE_DIR):
            logger.info("using the baseline in %s/baseline", self.uri)

    def push_params(self):
        upload(PARAMS_FILE, f"{self.run}/params.yaml")

    def push_lock(self, stage, attempts=20):
        """Merge this stage's entry into the shared dvc.lock.

        Parallel tasks (for example bias and preprocess) finish at about the
        same time, each in its own pod with its own copy of dvc.lock. A plain
        upload would let the last one overwrite the other's entry, so the
        write is conditional on the object's generation and retried.
        """
        own = yaml.safe_load(LOCK_FILE.read_text()) if LOCK_FILE.exists() \
            else {}
        if stage not in (own.get("stages") or {}):
            return
        uri = f"{self.uri}/dvc.lock"
        for _ in range(attempts):
            shared, generation = read_versioned(uri)
            merged = merge_lock_entry(shared, own, stage)
            if write_if_generation(uri, merged, generation):
                LOCK_FILE.write_text(merged)
                upload(LOCK_FILE, f"{self.run}/dvc.lock")
                logger.info("merged stage %s into %s", stage, uri)
                return
            time.sleep(random.uniform(0.2, 1.0))  # another stage wrote it
        raise RuntimeError(f"could not update {uri} after {attempts} tries")

    def push_artifacts(self, stage):
        for d in UPLOAD_DIRS:
            if (ROOT / d).exists():
                upload_tree(ROOT / d, f"{self.run}/{stage}/{d}")


# ---------------------------------------------------------------- stages
def open_state():
    """The run's GCS state; fails early without a real DVC remote."""
    if not remote_url():
        raise RuntimeError("no DVC remote configured; see README 'DVC remote'")
    uri = os.environ.get("PIPELINE_STATE_URI") or DEFAULT_STATE_URI
    return State(uri, run_id())


def run_stage(stage, overrides=()):
    if stage not in stages():
        raise ValueError(f"unknown stage {stage!r}; stages: {list(stages())}")
    state = open_state()
    state.pull(bool(overrides))
    set_params(overrides)
    if overrides:
        state.push_params()

    upstream = produced_by()
    for dep in stage_deps(stage):
        if upstream.get(dep, stage) != stage:
            # missing data is reported by repro below
            dvc("pull", dep, check=False)
    rc = 1
    try:
        # --pull: an unchanged stage restores its outputs from the remote
        # instead of running again
        rc = dvc("repro", "--single-item", "--pull", stage, check=False)
        if rc == 0:
            dvc("push", stage)
            state.push_lock(stage)
    finally:
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
    state = open_state()
    state.pull(False)
    for path in produced_by():
        dvc("pull", path, check=False)
    dvc("push")
    record = version_record()
    versions = LOG_DIR / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    path = versions / f"{record['snapshot']}__{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2))
    logger.info("data version %s: %d outputs, validation %s", path.name,
                len(record["outputs"]),
                (record["validation"] or {}).get("success"))
    base = f"{state.uri}/versions/{record['snapshot']}"
    upload(path, f"{base}/{record['run_id']}.json")
    upload(LOCK_FILE, f"{base}/dvc.lock")
    upload(LOCK_FILE, f"{base}/latest.dvc.lock")
    return record


def freeze(reason):
    """Make the latest schema and statistics the baseline, in GCS."""
    from scripts.schema_stats import freeze as freeze_files
    state = open_state()
    state.pull(False)
    for out in ("schema/generated/expectation_suite.json",
                "reports/statistics/statistics.json"):
        dvc("pull", out)
    info = freeze_files(reason)
    upload_tree(BASELINE_DIR, f"{state.uri}/baseline")
    return info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage",
                        help="a stage in dvc.yaml, publish, or freeze")
    parser.add_argument("--param", action="append", default=[],
                        help="override params.yaml, e.g. "
                             "acquire.snapshot=20261001 (repeatable)")
    parser.add_argument("--reason", default="",
                        help="freeze: why this snapshot is the baseline")
    args = parser.parse_args()
    setup_logging(f"run_stage_{args.stage}")
    try:
        if args.stage == "publish":
            publish()
        elif args.stage == "freeze":
            freeze(args.reason)
        else:
            run_stage(args.stage, args.param)
    except Exception:
        logger.exception("run_stage %s failed", args.stage)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
