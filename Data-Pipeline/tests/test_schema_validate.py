import copy
import json

import numpy as np
import pytest

from scripts import schema_stats
from scripts.schema_stats import (column_kind, compute_statistics,
                                  infer_suite)
from scripts.validate import (compare_statistics, describe, run_suite,
                              summarize)


def failed(checks, severity=None):
    return [c for c in checks if not c["success"]
            and (severity is None or c["severity"] == severity)]


# ---------------------------------------------------------------- statistics
def test_column_kinds(toy):
    kinds = {c: column_kind(c, toy[c]) for c in toy.columns}
    assert kinds == {"nct_id": "id", "overall_status": "categorical",
                     "label": "label", "start_date": "date",
                     "study_first_submitted_date": "date",
                     "has_dmc": "binary", "n_facilities": "numeric",
                     # a count that is only 0/1 in this snapshot stays numeric
                     "n_other_outcomes": "numeric"}


def test_statistics_values(toy):
    s = compute_statistics(toy, {"snapshot": "x"})
    assert s["snapshot"] == "x"
    assert s["dataset"]["n_rows"] == 200
    assert s["dataset"]["label_counts"] == {"stopped": 100, "completed": 100,
                                            "unfinished": 0}
    assert s["dataset"]["stopped_rate"] == 0.5
    n = s["columns"]["n_facilities"]
    assert (n["min"], n["max"], n["mean"]) == (0.0, 199.0, 99.5)
    assert n["quantiles"]["p50"] == 99.5
    assert s["columns"]["overall_status"]["top_values"] == {
        "COMPLETED": 100, "TERMINATED": 100}
    assert s["columns"]["nct_id"]["duplicates"] == 0
    assert s["columns"]["start_date"]["invalid_rate"] == 0.0


def test_statistics_count_missing(toy):
    toy.loc[:49, "n_facilities"] = np.nan
    c = compute_statistics(toy)["columns"]["n_facilities"]
    assert c["missing"] == 50 and c["missing_rate"] == 0.25


def test_statistics_count_outliers(toy):
    c = compute_statistics(toy)["columns"]["n_facilities"]
    assert c["outliers"]["count"] == 0  # 0..199, evenly spread
    toy.loc[:4, "n_facilities"] = 5000
    c = compute_statistics(toy)["columns"]["n_facilities"]
    assert c["outliers"]["count"] == 5 and c["outliers"]["rate"] == 0.025
    assert c["outliers"]["high"] < 5000


def test_filled_values_are_not_outliers(toy):
    toy.loc[:4, "n_facilities"] = 5000
    toy["n_facilities_missing"] = 0
    toy.loc[:4, "n_facilities_missing"] = 1  # filled in preprocessing
    c = compute_statistics(toy)["columns"]["n_facilities"]
    assert c["outliers"]["count"] == 0


def test_statistics_and_anomalies_use_one_outlier_rule(toy):
    from scripts import anomalies
    toy.loc[:19, "n_facilities"] = 5000  # 10% outliers, above the 5% limit
    stats = compute_statistics(toy, outlier_k=3.0)["columns"]["n_facilities"]
    found = [f for f in anomalies.outlier_anomalies(toy, 0.05, 3.0)
             if f["column"] == "n_facilities"]
    assert found and found[0]["count"] == stats["outliers"]["count"] == 20


# ---------------------------------------------------------------- schema
def test_suite_has_the_critical_contract(toy):
    suite = infer_suite(compute_statistics(toy))
    by_type = {}
    for e in suite.expectations:
        by_type.setdefault(e.expectation_type, []).append(e)
    table = by_type["expect_table_columns_to_match_set"][0]
    assert set(table.column_set) == set(toy.columns)
    assert table.severity.value == "critical"
    unique = [e.column for e in by_type["expect_column_values_to_be_unique"]]
    assert unique == ["nct_id"]
    ranges = by_type["expect_column_values_to_be_between"]
    assert all(e.severity.value == "warning" for e in ranges)


def test_clean_snapshot_passes(features):
    suite = infer_suite(compute_statistics(features))
    checks = run_suite(features, suite)
    assert failed(checks) == []


def test_breaking_changes_are_critical(features):
    suite = infer_suite(compute_statistics(features))
    bad = features.drop(columns=["n_arms"])
    bad.loc[1, "nct_id"] = bad.loc[0, "nct_id"]
    bad.loc[2, "label"] = 7
    bad.loc[3, "nct_id"] = "BAD-ID"
    checks = run_suite(bad, suite)
    critical = {(c["expectation"], c["column"])
                for c in failed(checks, "critical")}
    assert ("column_present", "n_arms") in critical
    assert ("expect_table_columns_to_match_set", None) in critical
    assert ("expect_column_values_to_be_unique", "nct_id") in critical
    assert ("expect_column_values_to_be_in_set", "label") in critical
    assert ("expect_column_values_to_match_regex", "nct_id") in critical
    assert not any(c.get("error") for c in checks)  # nothing errored


def test_out_of_range_values_are_warnings(features):
    suite = infer_suite(compute_statistics(features))
    bad = features.copy()
    bad.loc[:9, "n_facilities"] = 10_000
    checks = failed(run_suite(bad, suite))
    assert [(c["expectation"], c["column"], c["severity"]) for c in checks] \
        == [("expect_column_values_to_be_between", "n_facilities", "warning")]
    assert "10000" in describe(checks[0])


# ---------------------------------------------------------------- drift
def test_drift_findings(features):
    base = compute_statistics(features)
    same = compare_statistics(base, copy.deepcopy(base))
    assert same and all(c["success"] for c in same)

    drifted = features.copy()
    drifted["n_facilities"] = drifted["n_facilities"] * 5 + 50
    drifted.loc[:, "label"] = 1.0
    findings = failed(compare_statistics(compute_statistics(drifted), base))
    names = {(c["expectation"], c["column"]) for c in findings}
    assert ("mean_shift_std", "n_facilities") in names
    assert ("stopped_rate_change", "label") in names
    assert all(c["severity"] == "warning" for c in findings)


def test_row_count_drop_is_flagged(features):
    base = compute_statistics(features)
    small = compute_statistics(features.head(50))
    names = [c["expectation"] for c in failed(compare_statistics(small, base))]
    assert "row_count_change" in names


# ---------------------------------------------------------------- summary
@pytest.mark.parametrize("fail_on, stop", [("critical", False),
                                           ("warning", True),
                                           ("never", False)])
def test_fail_on_policy_for_warnings(fail_on, stop):
    checks = [{"success": True, "severity": "critical"},
              {"success": False, "severity": "warning"}]
    s = summarize(checks, "baseline", "20261001", fail_on)
    assert s["failed_warning"] == 1 and s["success"] is True
    assert s["stop_pipeline"] is stop


def test_critical_failure_stops_the_pipeline():
    checks = [{"success": False, "severity": "critical"}]
    s = summarize(checks, "baseline", "20261001", "critical")
    assert s["success"] is False and s["stop_pipeline"] is True


def test_snapshot_info_comes_from_the_features_meta(tmp_path, monkeypatch):
    """A task has only its own inputs: no raw snapshot, but the features'
    meta.json, which preprocess fills from the snapshot's manifest."""
    meta = tmp_path / "structured_features.meta.json"
    meta.write_text(json.dumps({"snapshot": "20261101", "source": "db",
                                "builder": "structured_features"}))
    monkeypatch.setattr(schema_stats, "RAW_DIR", tmp_path / "no-raw")
    monkeypatch.setattr(schema_stats, "FEATURES_META", meta)
    assert schema_stats.snapshot_info() == {
        "snapshot": "20261101", "source": "db",
        "builder": "structured_features"}
