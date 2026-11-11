"""Stage load_data: load the exact data version the data pipeline published
and split it by time into train, validation and test sets.

source "lock" (default, used on Vertex AI): reads the snapshot's dvc.lock
from gs://<bucket>/data-pipeline/state/versions/<snapshot>/ and downloads
each file by its md5 from the DVC remote, so the model always records which
data version it was trained on. source "local": the files already pulled
with `dvc pull preprocess bias` in Data-Pipeline/.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import yaml

from model_pipeline import tracking

logger = logging.getLogger(__name__)
REQUIRED = ("features", "meta", "slices")


# ---------- finding files in a DVC data version ----------

def find_out(lock, wanted):
    """(md5, path inside a directory output or None) of the dvc.lock output
    that holds the file `wanted`; (None, None) if no output holds it."""
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs", []) or []:
            path = out["path"].rstrip("/")
            if path == wanted:
                return out["md5"], None
            if wanted.startswith(path + "/"):
                return out["md5"], wanted[len(path) + 1:]
    return None, None


def remote_paths(remote, md5):
    """Where DVC 3 (files/md5/) or DVC 2 (top level) keeps an object."""
    return [f"{remote}/files/md5/{md5[:2]}/{md5[2:]}",
            f"{remote}/{md5[:2]}/{md5[2:]}"]


def _fetch(fs, remote, md5, dest):
    for path in remote_paths(remote, md5):
        if fs.exists(path):
            fs.get(path, str(dest))
            return
    raise FileNotFoundError(f"object {md5} not in the DVC remote {remote}")


def fetch_from_lock(fs, remote, lock, wanted, dest):
    """Download one file of the data version; returns its md5 or None."""
    md5, inner = find_out(lock, wanted)
    if md5 is None:
        return None
    if inner is None:
        _fetch(fs, remote, md5, dest)
        return md5
    listing = dest.with_name(dest.name + ".dir.json")  # directory output
    _fetch(fs, remote, md5, listing)
    entries = json.loads(listing.read_text())
    listing.unlink()
    for entry in entries:
        if entry["relpath"] == inner:
            _fetch(fs, remote, entry["md5"], dest)
            return entry["md5"]
    return None


def latest_snapshot(fs, base):
    names = sorted(Path(p.rstrip("/")).name for p in fs.ls(base))
    snapshots = [n for n in names if n.isdigit()]
    if not snapshots:
        raise FileNotFoundError(f"no data versions in gs://{base}")
    return snapshots[-1]


def download(ctx, raw):
    """Get the data files into raw/; returns a data-version record."""
    cfg = ctx.params["data"]
    files = cfg["files"]
    if cfg["source"] == "local":
        base = (Path(__file__).resolve().parents[1] / cfg["local_dir"])
        hashes = {}
        for key, rel in files.items():
            src = base / rel
            if not src.exists():
                if key in REQUIRED:
                    raise FileNotFoundError(
                        f"{src} missing: run `dvc pull preprocess bias` "
                        "in Data-Pipeline/")
                continue
            shutil.copy(src, raw / Path(rel).name)
            hashes[key] = hashlib.md5(src.read_bytes()).hexdigest()
        return {"source": "local", "snapshot": "local", "md5": hashes}

    import gcsfs
    fs = gcsfs.GCSFileSystem()
    versions = f"{cfg['bucket']}/{cfg['versions_prefix']}"
    snapshot = str(cfg["snapshot"])
    if snapshot == "latest":
        snapshot = latest_snapshot(fs, versions)
    lock_path = f"{versions}/{snapshot}/dvc.lock"
    lock = yaml.safe_load(fs.cat(lock_path))
    remote = f"{cfg['bucket']}/{cfg['dvc_remote']}"
    hashes = {}
    for key, rel in files.items():
        md5 = fetch_from_lock(fs, remote, lock, rel, raw / Path(rel).name)
        if md5 is None and key in REQUIRED:
            raise FileNotFoundError(f"{rel} is not in gs://{lock_path}")
        hashes[key] = md5
    logger.info("data version %s from gs://%s", snapshot, lock_path)
    return {"source": "lock", "snapshot": snapshot,
            "dvc_lock": f"gs://{lock_path}", "md5": hashes}


# ---------- splitting ----------

def split_by_time(finished, val_share, test_share):
    """Oldest trials train, newer validation, newest test (by registration
    date, the same order the data pipeline's bias_model stage uses)."""
    from scripts.bias_model import time_split

    train_val, test = time_split(finished, test_share)
    train, val = time_split(train_val, val_share / (1 - test_share))
    return train, val, test


def run(ctx):
    from scripts.bias_model import feature_columns

    raw = ctx.path("data", "raw", ".keep").parent
    version = download(ctx, raw)

    features = pd.read_csv(raw / "structured_features.csv",
                           dtype={"nct_id": str})
    meta = json.loads((raw / "structured_features.meta.json").read_text())
    slices = pd.read_csv(raw / "trial_slices.csv", dtype={"nct_id": str})
    finished = features[features["label"].notna()].copy()
    finished["label"] = finished["label"].astype(int)
    if finished["label"].nunique() < 2:
        raise ValueError("finished trials have only one class")
    columns = feature_columns(finished, meta)

    split = ctx.params["split"]
    parts = dict(zip(("train", "val", "test"),
                     split_by_time(finished, split["val_share"],
                                   split["test_share"])))
    slice_names = [s for s in ctx.params["bias"]["slices"]
                   if s in slices.columns]
    weights_file = raw / "trial_weights.csv"
    weights = pd.read_csv(weights_file, dtype={"nct_id": str}) \
        if weights_file.exists() else pd.DataFrame(columns=["nct_id",
                                                            "weight"])
    summary = {}
    for name, df in parts.items():
        if df["label"].nunique() < 2:
            raise ValueError(f"the {name} set has only one class")
        rows = df[["nct_id", "label", "study_first_submitted_date",
                   *columns]].merge(
            slices[["nct_id", *slice_names]], on="nct_id", how="left")
        for s in slice_names:
            rows[s] = rows[s].fillna("UNKNOWN").astype(str)
        rows = rows.merge(weights[["nct_id", "weight"]], on="nct_id",
                          how="left")
        rows["weight"] = rows["weight"].fillna(1.0)
        rows.to_pickle(ctx.path("data", f"{name}.pkl"))
        dates = pd.to_datetime(rows["study_first_submitted_date"],
                               errors="coerce")
        summary[name] = {"n": len(rows),
                         "stopped_rate": float(rows["label"].mean()),
                         "first": str(dates.min().date()),
                         "last": str(dates.max().date())}

    version.update({"n_features": len(columns), "splits": summary,
                    "slices": slice_names})
    ctx.write_json("data/columns.json", columns)
    ctx.write_json("data/data_version.json", version)
    ctx.update_state(snapshot=version["snapshot"])

    tracking.log_params(ctx, {"data_snapshot": version["snapshot"],
                              "data_source": version["source"],
                              "features_md5": version["md5"]["features"],
                              "n_features": len(columns),
                              **{f"n_{k}": v["n"]
                                 for k, v in summary.items()}})
    logger.info("splits: %s", {k: v["n"] for k, v in summary.items()})
    return version


def load_split(ctx, name):
    return pd.read_pickle(ctx.path("data", f"{name}.pkl"))


def load_columns(ctx):
    return ctx.read_json("data/columns.json")
