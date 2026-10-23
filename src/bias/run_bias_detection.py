"""Run bias detection: load trials, build slices, write the reports.

    python -m src.bias.run_bias_detection --config configs/bias_config.yaml

Writes to report.output_dir:
  data_slices.csv        size and stopped rate of every group
  data_summary.json      overall numbers and the flagged groups
  model_slices.csv       model metrics per group (only if predictions given)
  model_gaps.csv         best minus worst group per metric (same)
and the trials with their slice columns to report.slices_path.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.bias.load_data import load_trials
from src.bias.model_slices import model_slice_report
from src.bias.slice_report import data_slice_report
from src.bias.slicing import add_slices

logger = logging.getLogger("bias_detection")


def setup_logging(log_dir="logs"):
    """Log to the console and to logs/bias_detection.log."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(Path(log_dir) / "bias_detection.log")],
        force=True)


def run_model_report(trials, slice_names, model_config, out_dir):
    """Join the model scores to the finished trials and report per group."""
    preds = pd.read_csv(model_config["predictions_path"])
    finished = trials[trials.label.notna()]
    scored = finished.merge(preds[["nct_id", "y_score"]], on="nct_id")
    logger.info("model report on %d scored trials", len(scored))
    table, gaps = model_slice_report(scored, slice_names,
                                     model_config["threshold"],
                                     model_config["min_group_size"])
    table.to_csv(out_dir / "model_slices.csv", index=False)
    gaps.to_csv(out_dir / "model_gaps.csv", index=False)
    for g in gaps.itertuples():
        logger.info("%s: PR-AUC gap %.3f, recall gap %.3f", g.slice,
                    g.pr_auc, g.recall)


def run(config):
    trials = load_trials(config["input"])
    sliced = add_slices(trials, config["slices"])
    slice_names = config["slices"]["names"]

    slices_path = Path(config["report"]["slices_path"])
    slices_path.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(slices_path, index=False)
    logger.info("wrote trial slices to %s", slices_path)

    out_dir = Path(config["report"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report, summary = data_slice_report(sliced, slice_names,
                                        config["report"]["min_share"],
                                        config["report"]["max_rate_gap"])
    report.to_csv(out_dir / "data_slices.csv", index=False)
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("finished trials: %d, unfinished (left out): %d, "
                "overall stopped rate: %.3f", summary["n_finished"],
                summary["n_unfinished"], summary["overall_stopped_rate"])
    for g in summary["flagged_groups"]:
        # %s, not %.3f: stopped_rate is None for groups with no finished trials
        logger.warning("flagged %s=%s: share %s, stopped rate %s",
                       g["slice"], g["group"], g["share"],
                       g["stopped_rate"])

    if config["model"].get("predictions_path"):
        run_model_report(sliced, slice_names, config["model"], out_dir)
    logger.info("reports written to %s", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Bias detection by slicing")
    parser.add_argument("--config", default="configs/bias_config.yaml")
    args = parser.parse_args()

    setup_logging()
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        run(config)
    except Exception:
        logger.exception("bias detection failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
