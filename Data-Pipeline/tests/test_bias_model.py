"""Baseline model stage: features, time split, per-group report."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import bias_model

CONFIG = {"slices": {"names": ["phase", "region", "study_type"]},
          "model": {"threshold": 0.5, "min_group_size": 20}}


@pytest.fixture()
def stage_files(features, tmp_path, monkeypatch):
    """Features, their meta, trial slices and mitigation weights on disk."""
    rng = np.random.default_rng(0)
    finished = features[features["label"].notna()]
    slices = pd.DataFrame({
        "nct_id": features["nct_id"],
        "phase": rng.choice(["PHASE1", "PHASE2", "PHASE3"], len(features)),
        "region": rng.choice(["US_ONLY", "NON_US_ONLY", "US|NON_US"],
                             len(features)),
        "study_type": "INTERVENTIONAL",
    })
    weights = pd.DataFrame({"nct_id": finished["nct_id"],
                            "weight": np.where(np.arange(len(finished)) % 5,
                                               1.0, 4.0)})
    paths = {"FEATURES_CSV": tmp_path / "features.csv",
             "FEATURES_META": tmp_path / "features.meta.json",
             "TRIAL_SLICES": tmp_path / "trial_slices.csv",
             "WEIGHTS": tmp_path / "trial_weights.csv",
             "BIAS_MODEL_DIR": tmp_path / "bias_model"}
    features.to_csv(paths["FEATURES_CSV"], index=False)
    paths["FEATURES_META"].write_text(json.dumps(
        {"leakage_risk": ["n_facilities"]}))
    slices.to_csv(paths["TRIAL_SLICES"], index=False)
    weights.to_csv(paths["WEIGHTS"], index=False)
    for name, path in paths.items():
        monkeypatch.setattr(bias_model, name, path)
    return paths


def test_features_leave_out_ids_dates_label_and_leakage(features):
    columns = bias_model.feature_columns(
        features, {"leakage_risk": ["n_facilities"]})
    assert columns
    assert not {"nct_id", "label", "overall_status", "start_date",
                "study_first_submitted_date", "n_facilities"} & set(columns)


def test_time_split_tests_on_the_most_recent_trials():
    df = pd.DataFrame({
        "nct_id": [f"NCT{i:08d}" for i in range(10)],
        "study_first_submitted_date": [f"20{10 + i}-01-01"
                                       for i in range(10)][::-1],
        "label": [0, 1] * 5})
    train, test = bias_model.time_split(df, 0.2)
    assert len(test) == 2 and len(train) == 8
    assert set(test["study_first_submitted_date"]) == {"2018-01-01",
                                                       "2019-01-01"}


def test_report_covers_every_group_for_both_models(stage_files):
    summary = bias_model.run({"test_share": 0.3, "min_stopped": 1}, CONFIG)
    out = stage_files["BIAS_MODEL_DIR"]
    table = pd.read_csv(out / "model_slices.csv")
    gaps = pd.read_csv(out / "model_gaps.csv")
    assert set(table["model"]) == {"baseline", "reweighted",
                                   "group_thresholds"}
    # study_type has one value among finished trials: not a slice here
    assert set(gaps["slice"]) == {"phase", "region"}
    # a trial in "US|NON_US" counts in both region groups
    regions = set(table.loc[table["slice"] == "region", "group"])
    assert {"US", "NON_US", "US_ONLY", "NON_US_ONLY"} <= regions
    assert summary["upweighted_trials"] > 0
    assert set(summary["group_thresholds"]) <= {"US_ONLY", "NON_US_ONLY",
                                                "US|NON_US"}
    assert summary["n_test"] == round(0.3 * (summary["n_train"]
                                             + summary["n_test"]))
    for m in summary["overall"].values():
        assert 0 <= m["recall"] <= 1 and 0 <= m["flagged_rate"] <= 1
    assert "## Gap between the best and the worst group" in \
        (out / "bias_model.md").read_text()


def test_gaps_above_the_limit_are_flagged(stage_files):
    summary = bias_model.run({"test_share": 0.3, "max_gap": -1,
                              "min_stopped": 1}, CONFIG)
    flagged = {(f["model"], f["slice"]) for f in summary["flagged"]}
    assert ("baseline", "phase") in flagged


def test_one_class_cannot_be_trained(stage_files, features):
    one = features.assign(label=features["label"].where(
        features["label"].isna(), 0.0))
    one.to_csv(stage_files["FEATURES_CSV"], index=False)
    with pytest.raises(ValueError, match="one class"):
        bias_model.run({}, CONFIG)


def test_group_thresholds_equalize_recall():
    """Group b's stopped trials score lower; its own threshold gives it the
    same recall as the model overall."""
    rng = np.random.default_rng(1)
    groups = np.array(["a"] * 500 + ["b"] * 500)
    labels = rng.integers(0, 2, 1000)
    scores = np.clip(0.3 * labels + rng.uniform(0, 0.6, 1000)
                     - 0.2 * (groups == "b"), 0, 1)
    cuts = bias_model.group_thresholds(scores, labels, groups, 0.5, 10)
    shifted = bias_model.shift_scores(scores, groups, cuts, 0.5)
    recall = {g: np.mean(shifted[(groups == g) & (labels == 1)] >= 0.5)
              for g in "ab"}
    before = {g: np.mean(scores[(groups == g) & (labels == 1)] >= 0.5)
              for g in "ab"}
    assert abs(before["a"] - before["b"]) > 0.3
    assert abs(recall["a"] - recall["b"]) < 0.05
    # the order of trials within a group does not change
    a = groups == "a"
    assert (np.argsort(shifted[a]) == np.argsort(scores[a])).all()


def test_unknown_and_small_groups_are_left_out_of_gaps():
    table = pd.DataFrame({
        "model": "baseline", "slice": "region",
        "group": ["US", "NON_US", "UNKNOWN", "TINY"],
        "n_trials": [500, 500, 500, 500],
        "stopped_rate": [0.2, 0.2, 0.2, 0.01],
        "pr_auc": [0.6, 0.5, 0.99, 0.0], "recall": [0.8, 0.7, 0.9, 0.0],
        "precision": [0.3, 0.3, 0.9, 0.0],
        "flagged_rate": [0.5, 0.4, 0.2, 0.0]})
    (gap,) = bias_model.gaps_by_slice(table, 100, 10).itertuples()
    assert gap.n_groups_used == 2
    assert gap.pr_auc == pytest.approx(0.1)
