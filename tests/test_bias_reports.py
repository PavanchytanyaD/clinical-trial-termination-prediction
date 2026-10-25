import json

import numpy as np
import pandas as pd
import pytest

from src.bias.model_slices import model_slice_report
from src.bias.slice_report import data_slice_report


def sliced_trials():
    # 4 finished trials (2 stopped) and 1 unfinished trial.
    return pd.DataFrame({
        "nct_id": ["NCT1", "NCT2", "NCT3", "NCT4", "NCT5"],
        "label": [1.0, 0.0, 0.0, 1.0, np.nan],
        "phase": ["PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE3"],
        "therapeutic_area": ["neoplasms", "neoplasms|infections",
                             "infections", "UNKNOWN", "neoplasms"],
    })


def get_row(report, slice_name, group):
    rows = report[(report.slice == slice_name) & (report.group == group)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_data_report_counts_and_rates():
    report, summary = data_slice_report(sliced_trials(), ["phase"],
                                        min_share=0.3, max_rate_gap=0.4)
    assert summary["n_finished"] == 4
    assert summary["n_unfinished"] == 1
    assert summary["overall_stopped_rate"] == 0.5

    phase1 = get_row(report, "phase", "PHASE1")
    assert phase1.n_finished == 2 and phase1.n_stopped == 1
    assert phase1.stopped_rate == 0.5 and phase1.share == 0.5

    phase3 = get_row(report, "phase", "PHASE3")
    assert phase3.n_finished == 1 and phase3.n_unfinished == 1
    assert phase3.unfinished_share == 0.5
    # Shares of a single-value slice add up to 1.
    assert report.share.sum() == pytest.approx(1.0)


def test_data_report_flags():
    report, summary = data_slice_report(sliced_trials(), ["phase"],
                                        min_share=0.3, max_rate_gap=0.4)
    # PHASE2: 1 of 4 finished trials, 0% stopped vs 50% overall.
    phase2 = get_row(report, "phase", "PHASE2")
    assert phase2.too_small and phase2.large_gap
    assert not get_row(report, "phase", "PHASE1").too_small
    flagged = {(g["slice"], g["group"]) for g in summary["flagged_groups"]}
    assert flagged == {("phase", "PHASE2"), ("phase", "PHASE3")}


def test_multi_value_slice_counts_trial_in_each_group():
    report, _ = data_slice_report(sliced_trials(), ["therapeutic_area"],
                                  min_share=0.0, max_rate_gap=1.0)
    assert get_row(report, "therapeutic_area", "neoplasms").n_finished == 2
    assert get_row(report, "therapeutic_area", "infections").n_finished == 2


def test_group_with_only_unfinished_trials():
    trials = sliced_trials()
    trials.loc[4, "phase"] = "PHASE4"
    report, _ = data_slice_report(trials, ["phase"], 0.0, 1.0)
    phase4 = get_row(report, "phase", "PHASE4")
    assert phase4.n_finished == 0
    assert np.isnan(phase4.stopped_rate)


def test_summary_is_valid_json():
    trials = sliced_trials()
    trials.loc[4, "phase"] = "PHASE4"
    _, summary = data_slice_report(trials, ["phase"], 0.3, 1.0)
    text = json.dumps(summary, allow_nan=False)  # fails on NaN
    phase4 = [g for g in json.loads(text)["flagged_groups"]
              if g["group"] == "PHASE4"]
    assert phase4[0]["stopped_rate"] is None


def scored_trials():
    # Group A: model ranks perfectly. Group B: ranks backwards.
    # Group C: one trial, all completed, so PR-AUC is undefined.
    return pd.DataFrame({
        "nct_id": [f"NCT{i}" for i in range(9)],
        "label": [1, 1, 0, 0, 1, 1, 0, 0, 0],
        "y_score": [0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9, 0.7],
        "group_col": ["A"] * 4 + ["B"] * 4 + ["C"],
    })


def test_model_report_per_group():
    table, gaps = model_slice_report(scored_trials(), ["group_col"],
                                     threshold=0.5, min_group_size=2)
    a = get_row(table, "group_col", "A")
    b = get_row(table, "group_col", "B")
    c = get_row(table, "group_col", "C")
    assert a.pr_auc == 1.0 and a.recall == 1.0
    assert b.recall == 0.0
    assert np.isnan(c.pr_auc)
    assert get_row(table, "overall", "ALL").n_trials == 9

    # Group C (1 trial) is below min_group_size, so only A and B count.
    gap = gaps.iloc[0]
    assert gap.n_groups_used == 2
    assert gap.recall == 1.0
    assert gap.pr_auc == pytest.approx(1.0 - b.pr_auc)
