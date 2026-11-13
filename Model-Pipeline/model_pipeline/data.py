"""Stage load_data: load the exact data version the data pipeline published
and split it by time into train, validation and test sets.

source "lock" (default, used on Vertex AI): reads the snapshot's dvc.lock
from gs://<bucket>/data-pipeline/state/versions/<snapshot>/ and downloads
each file by its md5 from the DVC remote, so the model always records which
data version it was trained on. source "local": the files already pulled
with `dvc pull preprocess bias` in Data-Pipeline/.

Text features: the BioBERT embeddings of the data version (one 768-number
vector per text field) are reduced with PCA, fitted on the training set
only, to data.embeddings.pca_components numbers per field (columns
txt_<field>_<i>). Only model families with text_features: true use them.

Stop dates: the AACT studies table gives each stopped trial's stop date and
status. They are kept beside the features for the lead_time stage and never
used as features (they are written after the trial).
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from model_pipeline import tracking

logger = logging.getLogger(__name__)
REQUIRED = ("features", "meta", "slices")
EMBEDDING_FILES = ("ids.txt", "meta.json")


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


def targets(cfg, raw):
    """{key: (path in dvc.lock, local file)} of every file to get. The
    embedding files go to raw/embeddings/."""
    out = {key: (rel, raw / Path(rel).name)
           for key, rel in cfg["files"].items()}
    emb = cfg.get("embeddings") or {}
    if emb.get("dir"):
        names = [*EMBEDDING_FILES, *(f"{f}.npy" for f in emb["fields"])]
        for name in names:
            out[f"embeddings/{name}"] = (f"{emb['dir']}/{name}",
                                         raw / "embeddings" / name)
    return out


def download(ctx, raw):
    """Get the data files into raw/; returns a data-version record."""
    cfg = ctx.params["data"]
    files = targets(cfg, raw)
    (raw / "embeddings").mkdir(parents=True, exist_ok=True)
    if cfg["source"] == "local":
        base = (Path(__file__).resolve().parents[1] / cfg["local_dir"])
        hashes = {}
        for key, (rel, dest) in files.items():
            src = base / rel
            if not src.exists():
                if key in REQUIRED:
                    raise FileNotFoundError(
                        f"{src} missing: run `dvc pull preprocess bias` "
                        "in Data-Pipeline/")
                continue
            shutil.copy(src, dest)
            hashes[key] = _md5(src)
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
    for key, (rel, dest) in files.items():
        md5 = fetch_from_lock(fs, remote, lock, rel, dest)
        if md5 is None and key in REQUIRED:
            raise FileNotFoundError(f"{rel} is not in gs://{lock_path}")
        hashes[key] = md5
    logger.info("data version %s from gs://%s", snapshot, lock_path)
    return {"source": "lock", "snapshot": snapshot,
            "dvc_lock": f"gs://{lock_path}", "md5": hashes}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------- text features and stop dates ----------

def load_embeddings(folder, fields):
    """(nct_ids, {field: float32 array}) or None if the version has none."""
    folder = Path(folder)
    if not (folder / "ids.txt").exists():
        return None
    ids = (folder / "ids.txt").read_text().split()
    arrays = {}
    for field in fields:
        path = folder / f"{field}.npy"
        if path.exists():
            arrays[field] = np.load(path).astype(np.float32)
    if not arrays:
        return None
    return ids, arrays


def text_features(parts, embeddings, n_components, seed):
    """Add txt_<field>_<i> columns to every split. PCA is fitted on the
    training split only; trials without embeddings get zero vectors.
    Returns (text columns, fitted reducers)."""
    from sklearn.decomposition import PCA

    ids, arrays = embeddings
    position = pd.Series(np.arange(len(ids)), index=ids)
    columns, reducers = [], {}
    rows_of = {name: position.reindex(df["nct_id"]).to_numpy()
               for name, df in parts.items()}
    for field, array in arrays.items():
        def vectors(name):
            pos = rows_of[name]
            out = np.zeros((len(pos), array.shape[1]), dtype=np.float32)
            found = ~np.isnan(pos)
            out[found] = array[pos[found].astype(int)]
            return out

        k = min(n_components, array.shape[1], len(parts["train"]))
        pca = PCA(n_components=k, random_state=seed).fit(vectors("train"))
        names = [f"txt_{field}_{i:02d}" for i in range(k)]
        for name, df in parts.items():
            parts[name] = pd.concat(
                [df, pd.DataFrame(pca.transform(vectors(name)), columns=names,
                                  index=df.index)], axis=1)
        columns += names
        reducers[field] = pca
        logger.info("%s: %d PCA components keep %.0f%% of the variance",
                    field, k, 100 * pca.explained_variance_ratio_.sum())
    return columns, reducers


def stop_dates(raw):
    """nct_id, stop_date, stop_date_source from the AACT studies table, or
    None without it. The stop date is the actual completion date, else the
    actual primary completion date, else the last update (withdrawn trials
    often have neither)."""
    if not (raw / "studies.txt").exists():
        return None
    from src.features.build_text_sample import read_chunks

    cols = ["nct_id", "completion_date", "completion_date_type",
            "primary_completion_date", "primary_completion_date_type",
            "last_update_submitted_date"]
    s = pd.concat(read_chunks(raw, "studies", cols), ignore_index=True)
    s = s.drop_duplicates("nct_id")
    choices = [
        ("completion", s.completion_date,
         s.completion_date_type.str.upper() == "ACTUAL"),
        ("primary_completion", s.primary_completion_date,
         s.primary_completion_date_type.str.upper() == "ACTUAL"),
        ("last_update", s.last_update_submitted_date, True),
    ]
    date = pd.Series(pd.NaT, index=s.index)
    source = pd.Series(None, index=s.index, dtype=object)
    for name, values, ok in choices:
        parsed = pd.to_datetime(values, errors="coerce")
        use = date.isna() & parsed.notna() & ok
        date[use], source[use] = parsed[use], name
    return pd.DataFrame({"nct_id": s.nct_id, "stop_date": date,
                         "stop_date_source": source})


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
    stops = stop_dates(raw)
    info = ["overall_status"] if "overall_status" in finished.columns else []
    for name, df in parts.items():
        if df["label"].nunique() < 2:
            raise ValueError(f"the {name} set has only one class")
        rows = df[["nct_id", "label", "study_first_submitted_date", *info,
                   *columns]].merge(
            slices[["nct_id", *slice_names]], on="nct_id", how="left")
        for s in slice_names:
            rows[s] = rows[s].fillna("UNKNOWN").astype(str)
        rows = rows.merge(weights[["nct_id", "weight"]], on="nct_id",
                          how="left")
        rows["weight"] = rows["weight"].fillna(1.0)
        if stops is not None:
            rows = rows.merge(stops, on="nct_id", how="left")
        parts[name] = rows

    text_columns, emb_cfg = [], ctx.params["data"].get("embeddings") or {}
    embeddings = load_embeddings(raw / "embeddings", emb_cfg.get("fields",
                                                                 []))
    if embeddings is not None:
        text_columns, reducers = text_features(
            parts, embeddings, int(emb_cfg.get("pca_components", 16)),
            int(ctx.params["seed"]))
        joblib.dump({"fields": list(reducers), "reducers": reducers,
                     "columns": text_columns},
                    ctx.path("data", "text_pca.joblib"))
    else:
        logger.warning("no embeddings in this data version; models with "
                       "text_features use the structured features only")

    summary = {}
    for name, rows in parts.items():
        rows.to_pickle(ctx.path("data", f"{name}.pkl"))
        dates = pd.to_datetime(rows["study_first_submitted_date"],
                               errors="coerce")
        summary[name] = {"n": len(rows),
                         "stopped_rate": float(rows["label"].mean()),
                         "first": str(dates.min().date()),
                         "last": str(dates.max().date())}

    version.update({"n_features": len(columns),
                    "n_text_features": len(text_columns),
                    "has_stop_dates": stops is not None,
                    "splits": summary, "slices": slice_names})
    ctx.write_json("data/columns.json", columns)
    ctx.write_json("data/text_columns.json", text_columns)
    ctx.write_json("data/data_version.json", version)
    ctx.update_state(snapshot=version["snapshot"])

    tracking.log_params(ctx, {"data_snapshot": version["snapshot"],
                              "data_source": version["source"],
                              "features_md5": version["md5"]["features"],
                              "n_features": len(columns),
                              "n_text_features": len(text_columns),
                              **{f"n_{k}": v["n"]
                                 for k, v in summary.items()}})
    logger.info("splits: %s", {k: v["n"] for k, v in summary.items()})
    return version


def load_split(ctx, name):
    return pd.read_pickle(ctx.path("data", f"{name}.pkl"))


def load_columns(ctx, candidate=None):
    """Feature columns: the structured ones, plus the text ones for a
    candidate trained with text_features."""
    columns = ctx.read_json("data/columns.json")
    if candidate and candidate.get("text_features"):
        columns = columns + load_text_columns(ctx)
    return columns


def load_text_columns(ctx):
    path = ctx.path("data", "text_columns.json")
    return json.loads(path.read_text()) if path.exists() else []
