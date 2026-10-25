import datetime as dt

import numpy as np
import pytest

from scripts import anomalies, bias
from scripts.config import RAW_DIR, TRIAL_SLICES

TODAY = dt.date(2026, 10, 25)


def checks(findings, severity=None):
    return {(f["check"], f["column"]) for f in findings
            if severity is None or f["severity"] == severity}


def test_clean_snapshot_has_no_critical_anomalies(features):
    assert checks(anomalies.detect(features, today=TODAY), "critical") == set()


def test_id_and_label_anomalies(features):
    bad = features.copy()
    bad.loc[1, "nct_id"] = bad.loc[0, "nct_id"]
    bad.loc[2, "nct_id"] = "BAD-ID"
    bad.loc[3, "label"] = 7
    found = checks(anomalies.detect(bad, today=TODAY), "critical")
    assert {("duplicate_ids", "nct_id"), ("invalid_ids", "nct_id"),
            ("invalid_labels", "label")} <= found


def test_impossible_values(features):
    bad = features.copy()
    bad.loc[0, "n_facilities"] = -3
    bad.loc[1, "start_year"] = 2150
    bad["min_age_years"] = 30.0
    bad["max_age_years"] = 60.0
    bad.loc[2, ["min_age_years", "max_age_years"]] = [70.0, 20.0]
    found = anomalies.detect(bad, today=TODAY)
    assert {("impossible_values", "n_facilities"),
            ("impossible_values", "start_year"),
            ("impossible_values", "min_age_years")} <= checks(found)


def test_missing_flag_hides_filled_values(features):
    # a filled 0 with its _missing flag set is not an impossible year
    bad = features.copy()
    bad.loc[0, ["start_year", "start_year_missing"]] = [0, 1]
    assert ("impossible_values", "start_year") not in checks(
        anomalies.detect(bad, today=TODAY))


def test_single_class_is_critical(features):
    bad = features.copy()
    bad.loc[bad.label.notna(), "label"] = 0.0
    assert ("single_class", "label") in checks(
        anomalies.detect(bad, today=TODAY), "critical")


def test_missing_values_and_outliers_are_warnings(features):
    bad = features.copy()
    bad["n_arms_missing"] = (np.arange(len(bad)) % 2).astype(np.int8)
    bad.loc[:39, "n_facilities"] = 100_000
    found = checks(anomalies.detect(bad, today=TODAY), "warning")
    assert ("missing_values", "n_arms") in found
    assert ("outliers", "n_facilities") in found


def test_invalid_dates_are_warnings(features):
    bad = features.copy()
    bad.loc[:20, "start_date"] = "next spring"
    assert ("invalid_dates", "start_date") in checks(
        anomalies.detect(bad, today=TODAY), "warning")


@pytest.mark.parametrize("fail_on, stop", [("critical", False),
                                           ("warning", True),
                                           ("never", False)])
def test_fail_on_policy(fail_on, stop):
    found = [anomalies.finding("outliers", "warning", "x", 3)]
    assert anomalies.summarize(found, 10, fail_on)["stop_pipeline"] is stop


def test_bias_config_points_at_the_pipeline_snapshot():
    config = bias.bias_config()
    assert config["input"] == {"source": "snapshot",
                               "snapshot_dir": str(RAW_DIR)}
    assert config["report"]["slices_path"] == str(TRIAL_SLICES)
    assert config["mitigation"]["input_path"] == str(TRIAL_SLICES)


def test_bias_runs_on_the_sample_snapshot(snapshot_dir, tmp_path):
    from src.bias import run_bias_detection, run_bias_mitigation
    config = bias.bias_config()
    config["input"]["snapshot_dir"] = str(snapshot_dir)
    config["report"]["output_dir"] = str(tmp_path / "bias")
    config["report"]["slices_path"] = str(tmp_path / "slices.csv")
    config["mitigation"]["input_path"] = str(tmp_path / "slices.csv")
    config["mitigation"]["output_dir"] = str(tmp_path / "bias" / "m")
    run_bias_detection.run(config)
    summary = run_bias_mitigation.run(config)
    assert (tmp_path / "bias" / "data_slices.csv").exists()
    assert summary["n_trials"] > 0
