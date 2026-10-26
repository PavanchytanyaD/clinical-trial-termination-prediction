"""Pipeline stage: data bias detection and mitigation (Tasks 4 and 5).

Runs the team's bias code in src/bias/ on the snapshot this pipeline
versions, so the slicing report always matches the data of the run:

1. detection (src/bias/run_bias_detection.py): slices finished trials by
   phase, study type, sponsor class, region, therapeutic area, gender, age
   group and start era, and flags groups that are too small or whose
   stopped-trial rate is far from the overall rate,
2. mitigation (src/bias/run_bias_mitigation.py): reweights (or resamples)
   under-represented groups and reports each group before and after.

Settings come from configs/bias_config.yaml at the repository root; this
stage only points the input at data/raw/aact_snapshot/ and the outputs at
Data-Pipeline/. The stage depends on the raw snapshot only, so Airflow runs it
in parallel with preprocessing.

    python -m scripts.bias
"""

import json
import logging

import yaml

from scripts.config import (BIAS_CONFIG, BIAS_DIR, RAW_DIR, TRIAL_SLICES,
                            load_params)
from scripts.tracking import fingerprint, tracked_step

logger = logging.getLogger("bias")


def bias_config(path=BIAS_CONFIG):
    """The team's bias config with this pipeline's input and output paths."""
    config = yaml.safe_load(path.read_text())
    config["input"] = {"source": "snapshot", "snapshot_dir": str(RAW_DIR)}
    config["report"]["output_dir"] = str(BIAS_DIR)
    config["report"]["slices_path"] = str(TRIAL_SLICES)
    config.setdefault("mitigation", {})
    config["mitigation"]["input_path"] = str(TRIAL_SLICES)
    config["mitigation"]["output_dir"] = str(BIAS_DIR / "mitigation")
    return config


def main():
    from src.bias import run_bias_detection, run_bias_mitigation

    with tracked_step("bias") as run:
        params = load_params().get("bias", {})
        config = bias_config()
        if params.get("mitigation_method"):
            config["mitigation"]["method"] = params["mitigation_method"]
        run_bias_detection.run(config)
        summary = json.loads((BIAS_DIR / "data_summary.json").read_text())
        mitigation = run_bias_mitigation.run(config)
        run["metrics"].update({
            "trials": summary["n_trials"],
            "finished": summary["n_finished"],
            "stopped_rate": summary["overall_stopped_rate"],
            "flagged_groups": len(summary["flagged_groups"]),
            "boosted_groups": len(mitigation.get("boosted_groups", [])),
        })
        run["outputs"]["reports/bias"] = fingerprint(BIAS_DIR)
        logger.info("bias report: %d groups flagged, %d boosted",
                    len(summary["flagged_groups"]),
                    len(mitigation.get("boosted_groups", [])))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
