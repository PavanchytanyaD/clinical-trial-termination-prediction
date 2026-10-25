"""Pipeline step 1, data acquisition (adapter around the ingestion code).

Writes one AACT snapshot to data/raw/aact_snapshot/ with a manifest.json.
The source is set in params.yaml (acquire.source):

  sample  placeholder: a small synthetic snapshot (scripts/sample_snapshot.py)
  zip     AACT monthly flat-file export, path or URL (acquire.zip)
  db      AACT cloud PostgreSQL (AACT_USER / AACT_PASSWORD in the environment)

zip and db call src/ingestion/fetch_aact.py (data acquisition is owned by
another workstream), so this step only fixes where the snapshot lands and
how it is checked. Every source must produce the 12 tables in REQUIRED_TABLES,
each with an nct_id column and at least one row.

    python -m scripts.acquire
"""

import datetime as dt
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from scripts import sample_snapshot
from scripts.config import RAW_DIR, REPO_ROOT, REQUIRED_TABLES, load_params
from scripts.tracking import fingerprint, tracked_step

logger = logging.getLogger("acquire")


def ingestion_module():
    """Import src.ingestion.fetch_aact from the repository root."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from src.ingestion import fetch_aact
    except ImportError as e:
        raise RuntimeError(
            "src/ingestion/fetch_aact.py is not in this checkout (it is on "
            "the nithish/s1-text-cleaning branch until merged). Use "
            "acquire.source: sample, or merge the ingestion code.") from e
    return fetch_aact


def check_tables(out_dir, tables=REQUIRED_TABLES):
    """Each table file exists, has an nct_id column and at least one row."""
    out_dir = Path(out_dir)
    sizes = {}
    for t in tables:
        path = out_dir / f"{t}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing table file: {path}")
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("|")
            first_row = f.readline()
        if "nct_id" not in header:
            raise ValueError(f"{path} has no nct_id column")
        if not first_row.strip():
            raise ValueError(f"{path} has no data rows")
        sizes[t] = path.stat().st_size
    return sizes


def acquire(params, out_dir=RAW_DIR):
    source = params.get("source", "sample")
    snapshot = str(params.get("snapshot")
                   or dt.date.today().strftime("%Y%m%d"))
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # one snapshot per run; DVC keeps the history
    out_dir.mkdir(parents=True)
    logger.info("acquiring AACT snapshot %s from %s into %s", snapshot,
                source, out_dir)

    if source == "sample":
        return sample_snapshot.write_snapshot(
            out_dir, snapshot, n_trials=int(params.get("sample_trials", 500)),
            seed=int(params.get("seed", 42)))

    fetch_aact = ingestion_module()
    if source == "zip":
        zip_src = params.get("zip")
        if not zip_src:
            raise ValueError("acquire.source is zip but acquire.zip is empty")
        zip_path = zip_src
        if re.match(r"https?://", zip_src):
            zip_path = fetch_aact.download(
                zip_src, out_dir / Path(zip_src.split("?")[0]).name)
        fetch_aact.extract_tables(zip_path, out_dir)
        if zip_path != zip_src:
            Path(zip_path).unlink()
    elif source == "db":
        fetch_aact.export_from_db(out_dir)
    else:
        raise ValueError(f"unknown acquire.source {source!r} "
                         "(use sample, zip or db)")
    manifest = {
        "source": source if source == "db" else params.get("zip"),
        "snapshot": snapshot,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "tables": fetch_aact.validate(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    with tracked_step("acquire") as run:
        manifest = acquire(load_params().get("acquire", {}))
        sizes = check_tables(RAW_DIR)
        run["metrics"].update({"snapshot": manifest["snapshot"],
                               "source": manifest["source"],
                               "tables": len(sizes),
                               "raw_bytes": sum(sizes.values())})
        run["outputs"]["data/raw/aact_snapshot"] = fingerprint(RAW_DIR)
        logger.info("snapshot %s ready: %d tables, %d bytes",
                    manifest["snapshot"], len(sizes), sum(sizes.values()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
