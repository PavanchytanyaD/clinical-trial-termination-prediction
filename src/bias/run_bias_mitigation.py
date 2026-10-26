"""Main script for bias mitigation (guideline section 3, part 3).

Reads the per-trial slice table from bias detection, boosts small groups
(reweight or resample), and writes:
  trial_weights.csv            nct_id, weight   (for the training step)
  resampled_trials.csv         only when method is resample
  mitigation_before_after.csv  every group, before vs after
  mitigation_summary.json      what changed, for the report and alerts

run(config) is the function the Airflow DAG calls.
Run on its own with:  python -m src.bias.run_bias_mitigation --config configs/bias_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.bias.mitigate import (
    before_after,
    compute_weights,
    effective_sample_size,
    labeled_trials,
    overall_stopped_rate,
    resample,
)

logger = logging.getLogger("bias_mitigation")

# Used when a setting is missing from the config's "mitigation" section.
DEFAULTS = {
    "input_path": "data/processed/trial_slices.csv",
    "output_dir": "reports/bias/mitigation",
    "id_column": "nct_id",
    "label_column": "label",
    "multi_value_sep": "|",
    "slices": [
        "phase",
        "study_type",
        "sponsor_class",
        "gender",
        "therapeutic_area",
        "age_group",
        "region",
        "start_era",
    ],
    "skip_groups": ["UNKNOWN"],
    "method": "reweight",
    "min_share": 0.01,
    "max_weight": 10.0,
    "min_group_size": 100,
    "random_seed": 42,
}


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_slices(path: str | Path, id_col: str) -> pd.DataFrame:
    """Read the slice table (CSV or Parquet) and drop duplicate trials."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"slice table not found: {path}. Run bias detection first.")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        # keep AACT's literal "NA" (phase not applicable) as a value
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    else:
        raise ValueError(f"unsupported file type: {path.suffix} (use .csv or .parquet)")
    if id_col not in df.columns:
        raise KeyError(f"id column '{id_col}' not found in {path}")
    dupes = int(df[id_col].duplicated().sum())
    if dupes:
        logger.warning("dropping %d duplicate trials", dupes)
        df = df.drop_duplicates(subset=id_col)
    return df


def _clean(value):
    """Make numbers JSON-safe (NaN becomes null)."""
    if isinstance(value, float) and value != value:
        return None
    return value


def run(config: dict) -> dict:
    cfg = {**DEFAULTS, **(config.get("mitigation") or {})}
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    label, sep, slices = cfg["label_column"], cfg["multi_value_sep"], cfg["slices"]

    df = read_slices(cfg["input_path"], cfg["id_column"])
    df = labeled_trials(df, label)
    logger.info("loaded %d finished trials", len(df))

    weights, boosted, too_few = compute_weights(
        df,
        slices,
        min_share=cfg["min_share"],
        max_weight=cfg["max_weight"],
        sep=sep,
        skip_groups=tuple(cfg["skip_groups"]),
        min_group_size=cfg["min_group_size"],
    )
    logger.info("boosted %d small groups", len(boosted))
    for g in too_few:
        logger.warning(
            "too few trials to mitigate: %s=%s (%d trials)", g["slice"], g["group"], g["n_trials"]
        )

    pd.DataFrame({cfg["id_column"]: df[cfg["id_column"]], "weight": weights}).to_csv(
        out_dir / "trial_weights.csv", index=False
    )

    method = cfg["method"]
    if method == "reweight":
        table = before_after(df, df, slices, label, cfg["min_share"], sep, after_weights=weights)
        rate_after = overall_stopped_rate(df, label, weights)
        ess = effective_sample_size(weights)
    elif method == "resample":
        sampled = resample(df, weights, seed=cfg["random_seed"])
        sampled.to_csv(out_dir / "resampled_trials.csv", index=False)
        table = before_after(df, sampled, slices, label, cfg["min_share"], sep)
        rate_after = overall_stopped_rate(sampled, label)
        ess = effective_sample_size(weights)
        logger.info("resample kept %d unique trials", sampled[cfg["id_column"]].nunique())
    else:
        raise ValueError(f"unknown mitigation method '{method}' (use reweight or resample)")

    table.to_csv(out_dir / "mitigation_before_after.csv", index=False)

    # Groups we tried to boost but the weight cap stopped. Too-few groups are reported separately.
    too_few_keys = {(g["slice"], g["group"]) for g in too_few}
    still_small = table[
        table["too_small_after"]
        & ~table["group"].isin(cfg["skip_groups"])
        & ~pd.Series(list(zip(table["slice"], table["group"]))).isin(too_few_keys)
    ]
    for row in still_small.itertuples():
        logger.warning(
            "still too small after mitigation: %s=%s (share %.4f)",
            row.slice, row.group, row.share_after
        )

    summary = {
        "method": method,
        "n_trials": int(len(df)),
        "effective_sample_size": round(ess, 1),
        "overall_stopped_rate_before": round(overall_stopped_rate(df, label), 4),
        "overall_stopped_rate_after": round(rate_after, 4),
        "max_weight_used": round(float(weights.max()), 3),
        "min_group_size": cfg["min_group_size"],
        "boosted_groups": boosted,
        "too_few_to_mitigate": too_few,
        "still_too_small": [
            {"slice": r.slice, "group": r.group, "share_after": _clean(float(r.share_after))}
            for r in still_small.itertuples()
        ],
    }
    with (out_dir / "mitigation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    logger.info("wrote outputs to %s", out_dir)
    return summary


def _setup_logging(log_path: str = "logs/bias_mitigation.log") -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bias mitigation on the slice table.")
    parser.add_argument("--config", default="configs/bias_config.yaml")
    args = parser.parse_args()
    _setup_logging()
    try:
        run(load_config(args.config))
    except Exception:
        logger.exception("bias mitigation failed")
        sys.exit(1)  # non-zero exit so Airflow marks the task as failed


if __name__ == "__main__":
    main()
