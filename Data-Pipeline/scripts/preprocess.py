"""Pipeline step 2, preprocessing (adapter around the feature code).

Reads the snapshot in data/raw/aact_snapshot/ and writes one row per trial to
data/processed/structured_features.csv, plus a .meta.json. preprocess.mode in
params.yaml picks the builder:

  structured_features  src/features/structured_features.py (owned by another
                       workstream): cleaning, leakage exclusions and about
                       150 design features
  placeholder          a small stand-in built here, so the later steps can be
                       built and tested before the real code is merged
  auto                 structured_features when it is importable, else
                       placeholder

Both builders share the output contract the later steps rely on: columns
nct_id, overall_status, label (1 stopped, 0 completed, empty while running),
start_date, study_first_submitted_date, then numeric features.

    python -m scripts.preprocess
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from scripts.config import (FEATURES_CSV, FEATURES_META, RAW_DIR, REPO_ROOT,
                            load_params)
from scripts.tracking import fingerprint, tracked_step

logger = logging.getLogger("preprocess")

STOPPED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
ID_COLUMNS = ["nct_id", "overall_status", "label", "start_date",
              "study_first_submitted_date"]
PLACEHOLDER_PHASES = ["NA", "EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2",
                      "PHASE2", "PHASE2/PHASE3", "PHASE3", "PHASE4"]


def feature_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.features import structured_features
    return structured_features


def label_from_status(status):
    """1 = stopped early, 0 = completed, NaN = not finished (censored)."""
    return status.map(lambda s: 1.0 if s in STOPPED_STATUSES
                      else (0.0 if s == "COMPLETED" else np.nan))


def _read(export_dir, table, cols):
    return pd.read_csv(export_dir / f"{table}.txt", sep="|", usecols=cols,
                       dtype=str, keep_default_na=False)


def placeholder_features(export_dir=RAW_DIR):
    """A few design features, enough to exercise schema and validation."""
    studies = _read(export_dir, "studies",
                    ["nct_id", "overall_status", "study_type", "phase",
                     "number_of_arms", "has_dmc", "start_date",
                     "study_first_submitted_date"]).drop_duplicates("nct_id")
    calc = _read(export_dir, "calculated_values",
                 ["nct_id", "number_of_facilities",
                  "number_of_primary_outcomes_to_measure"])
    calc = calc.drop_duplicates("nct_id").set_index("nct_id")
    s = studies.set_index("nct_id")
    out = pd.DataFrame(index=s.index)
    out["overall_status"] = s.overall_status
    out["label"] = label_from_status(s.overall_status)
    out["start_date"] = s.start_date
    out["study_first_submitted_date"] = s.study_first_submitted_date
    out["study_type_interventional"] = (s.study_type == "INTERVENTIONAL"
                                        ).astype(np.int8)
    for p in PLACEHOLDER_PHASES:
        out["phase_" + p.lower().replace("/", "_")] = (s.phase == p
                                                       ).astype(np.int8)
    out["phase_missing"] = (s.phase == "").astype(np.int8)
    out["n_arms"] = pd.to_numeric(s.number_of_arms, errors="coerce")
    out["has_dmc"] = s.has_dmc.map({"t": 1.0, "f": 0.0})
    out["start_year"] = pd.to_numeric(s.start_date.str[:4], errors="coerce")
    out["n_facilities"] = pd.to_numeric(
        calc.number_of_facilities.reindex(s.index), errors="coerce")
    out["n_primary_outcomes"] = pd.to_numeric(
        calc.number_of_primary_outcomes_to_measure.reindex(s.index),
        errors="coerce")
    for col in ["n_arms", "has_dmc", "start_year", "n_facilities"]:
        out[f"{col}_missing"] = out[col].isna().astype(np.int8)
        out[col] = out[col].fillna(0)
    out["n_primary_outcomes"] = out["n_primary_outcomes"].fillna(0)
    out = out.reset_index()
    features = [c for c in out.columns if c not in ID_COLUMNS]
    meta = {
        "export_dir": str(export_dir),
        "n_rows": int(len(out)),
        "n_features": len(features),
        "label_counts": {
            "stopped": int((out.label == 1).sum()),
            "completed": int((out.label == 0).sum()),
            "unfinished": int(out.label.isna().sum()),
        },
        "feature_columns": features,
    }
    return out, meta


def build(mode="auto", export_dir=RAW_DIR):
    """Returns (features DataFrame, meta dict, builder name)."""
    if mode in ("auto", "structured_features"):
        try:
            module = feature_module()
        except ImportError:
            if mode == "structured_features":
                raise RuntimeError(
                    "src/features/structured_features.py is not in this "
                    "checkout; use preprocess.mode: placeholder or auto")
            logger.warning("src.features.structured_features not found; "
                           "using the placeholder feature builder")
        else:
            out, meta = module.build_features(export_dir)
            return out, meta, "structured_features"
    if mode not in ("auto", "placeholder"):
        raise ValueError(f"unknown preprocess.mode {mode!r}")
    out, meta = placeholder_features(export_dir)
    return out, meta, "placeholder"


def check_contract(df):
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature table is missing columns {missing}")
    if df.nct_id.duplicated().any():
        raise ValueError("feature table has duplicate nct_id rows")


def main():
    with tracked_step("preprocess") as run:
        mode = load_params().get("preprocess", {}).get("mode", "auto")
        out, meta, builder = build(mode)
        check_contract(out)
        meta["builder"] = builder
        FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(FEATURES_CSV, index=False, float_format="%.6g")
        FEATURES_META.write_text(json.dumps(meta, indent=2))
        run["metrics"].update({"builder": builder, "rows": meta["n_rows"],
                               "features": meta["n_features"],
                               **{f"label_{k}": v for k, v in
                                  meta["label_counts"].items()}})
        run["outputs"]["data/processed/structured_features.csv"] = \
            fingerprint(FEATURES_CSV)
        logger.info("wrote %d trials x %d features (%s) to %s",
                    meta["n_rows"], meta["n_features"], builder, FEATURES_CSV)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
