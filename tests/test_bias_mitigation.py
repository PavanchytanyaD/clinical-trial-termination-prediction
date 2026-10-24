"""Tests for bias mitigation (src/bias/mitigate.py and run_bias_mitigation.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.bias.mitigate import (
    UNKNOWN,
    before_after,
    compute_weights,
    effective_sample_size,
    group_shares,
    labeled_trials,
    resample,
    split_values,
)
from src.bias.run_bias_mitigation import run


def make_trials(n_big=95, n_small=3, n_unknown=2):
    """Toy slice table: one big sponsor group, one small one, a few UNKNOWN."""
    sponsors = ["INDUSTRY"] * n_big + ["NIH"] * n_small + [None] * n_unknown
    n = len(sponsors)
    return pd.DataFrame(
        {
            "nct_id": [f"NCT{i:08d}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "sponsor_class": sponsors,
            "therapeutic_area": ["neoplasms"] * n,
        }
    )


# --- small helpers -----------------------------------------------------------


def test_labeled_trials_drops_unfinished():
    df = pd.DataFrame({"nct_id": ["a", "b", "c"], "label": [1, None, 0]})
    out = labeled_trials(df, "label")
    assert list(out["nct_id"]) == ["a", "c"]
    assert out["label"].dtype.kind == "i"


def test_labeled_trials_missing_label_column():
    with pytest.raises(KeyError):
        labeled_trials(pd.DataFrame({"nct_id": ["a"]}), "label")


def test_split_values_multi_and_missing():
    out = split_values(pd.Series(["neoplasms|infections", None, ""]))
    assert out.tolist() == [["neoplasms", "infections"], [UNKNOWN], [UNKNOWN]]


def test_group_shares_counts_multi_value_trial_in_each_group():
    df = pd.DataFrame({"area": ["a|b", "a", "c", "c"]})
    shares = group_shares(df, "area")
    assert shares["a"] == pytest.approx(0.5)
    assert shares["b"] == pytest.approx(0.25)


# --- weights -----------------------------------------------------------------


def test_small_group_gets_higher_weight():
    df = make_trials()
    w, boosted, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=10)
    nih = df["sponsor_class"] == "NIH"
    assert w[nih].min() > w[~nih].max()
    assert [b["group"] for b in boosted] == ["NIH"]


def test_unknown_group_is_not_boosted():
    df = make_trials()
    w, boosted, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=10)
    unknown = df["sponsor_class"].isna()
    industry = df["sponsor_class"] == "INDUSTRY"
    assert w[unknown].max() == pytest.approx(w[industry].max())
    assert UNKNOWN not in [b["group"] for b in boosted]


def test_weights_have_mean_one_and_respect_cap():
    df = make_trials(n_big=995, n_small=1, n_unknown=0)
    w, _, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=5)
    assert w.mean() == pytest.approx(1.0)
    assert w.max() <= 5 + 1e-9
    assert (w > 0).all()


def test_no_small_groups_gives_equal_weights():
    df = make_trials(n_big=50, n_small=50, n_unknown=0)
    w, boosted, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=10)
    assert np.allclose(w, 1.0)
    assert boosted == []


def test_multi_area_trial_keeps_largest_factor():
    areas = ["common"] * 96 + ["common|rare"] * 2 + ["common|rarer"]
    df = pd.DataFrame({"label": [0] * len(areas), "area": areas})
    w, _, _ = compute_weights(df, ["area"], min_share=0.05, max_weight=100)
    rare = w[df["area"] == "common|rare"].iloc[0]
    rarer = w[df["area"] == "common|rarer"].iloc[0]
    assert rarer > rare > w[df["area"] == "common"].iloc[0]


def test_weight_cap_can_leave_group_below_target():
    df = make_trials(n_big=999, n_small=1, n_unknown=0)
    w, _, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=2)
    table = before_after(df, df, ["sponsor_class"], "label", 0.05, after_weights=w)
    nih = table[table["group"] == "NIH"].iloc[0]
    assert nih["share_after"] > nih["share_before"]
    assert nih["too_small_after"]  # the cap stopped it, and the report shows it


def test_two_slices_both_reach_target():
    n = 1000
    df = pd.DataFrame(
        {
            "label": [i % 2 for i in range(n)],
            "a": ["x"] * 995 + ["tiny"] * 5,
            "b": ["tiny"] * 5 + ["y"] * 995,
        }
    )
    w, _, _ = compute_weights(df, ["a", "b"], min_share=0.02, max_weight=10)
    table = before_after(df, df, ["a", "b"], "label", 0.02, after_weights=w)
    assert not table["too_small_after"].any()


def test_group_below_min_group_size_is_not_boosted():
    df = make_trials()  # NIH has 3 trials
    w, boosted, too_few = compute_weights(
        df, ["sponsor_class"], min_share=0.05, max_weight=10, min_group_size=5
    )
    assert np.allclose(w, 1.0)
    assert boosted == []
    assert too_few == [{"slice": "sponsor_class", "group": "NIH", "n_trials": 3, "share": 0.03}]


def test_group_at_min_group_size_is_boosted():
    df = make_trials()
    _, boosted, too_few = compute_weights(
        df, ["sponsor_class"], min_share=0.05, max_weight=10, min_group_size=3
    )
    assert [b["group"] for b in boosted] == ["NIH"]
    assert boosted[0]["n_trials"] == 3
    assert too_few == []


def test_negative_min_group_size_raises():
    with pytest.raises(ValueError):
        compute_weights(make_trials(), ["sponsor_class"], 0.05, 10, min_group_size=-1)


@pytest.mark.parametrize("min_share,max_weight", [(0, 10), (1.5, 10), (0.05, 0.5)])
def test_bad_settings_raise(min_share, max_weight):
    with pytest.raises(ValueError):
        compute_weights(make_trials(), ["sponsor_class"], min_share, max_weight)


def test_missing_slice_column_raises():
    with pytest.raises(KeyError):
        compute_weights(make_trials(), ["region"], 0.05, 10)


def test_effective_sample_size():
    assert effective_sample_size([1, 1, 1, 1]) == pytest.approx(4)
    assert effective_sample_size([4, 0, 0, 0]) == pytest.approx(1)


# --- before/after and resampling --------------------------------------------


def test_reweight_lifts_small_group_share():
    df = make_trials()
    w, _, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=10)
    table = before_after(df, df, ["sponsor_class"], "label", 0.05, after_weights=w)
    nih = table[table["group"] == "NIH"].iloc[0]
    assert nih["share_before"] == pytest.approx(0.03)
    assert nih["share_after"] > nih["share_before"]
    assert nih["too_small_before"] and not nih["too_small_after"]


def test_resample_same_size_and_reproducible():
    df = make_trials()
    w, _, _ = compute_weights(df, ["sponsor_class"], min_share=0.05, max_weight=10)
    a = resample(df, w, seed=1)
    b = resample(df, w, seed=1)
    assert len(a) == len(df)
    assert a["nct_id"].tolist() == b["nct_id"].tolist()


# --- end to end --------------------------------------------------------------


def _config(tmp_path, method="reweight"):
    data = make_trials()
    data.loc[0, "label"] = None  # one unfinished trial, should be dropped
    path = tmp_path / "data_slices.csv"
    data.to_csv(path, index=False)
    return {
        "mitigation": {
            "input_path": str(path),
            "output_dir": str(tmp_path / "out"),
            "slices": ["sponsor_class", "therapeutic_area"],
            "min_share": 0.05,
            "min_group_size": 0,
            "method": method,
        }
    }


@pytest.mark.parametrize("method", ["reweight", "resample"])
def test_run_writes_outputs(tmp_path, method):
    summary = run(_config(tmp_path, method))
    out = tmp_path / "out"
    assert (out / "trial_weights.csv").exists()
    assert (out / "mitigation_before_after.csv").exists()
    saved = json.loads((out / "mitigation_summary.json").read_text())
    assert saved["method"] == method
    assert saved["n_trials"] == 99
    assert summary["boosted_groups"][0]["group"] == "NIH"


def test_run_reports_too_few_groups(tmp_path):
    cfg = _config(tmp_path)
    cfg["mitigation"]["min_group_size"] = 10
    summary = run(cfg)
    assert summary["boosted_groups"] == []
    assert [g["group"] for g in summary["too_few_to_mitigate"]] == ["NIH"]
    assert summary["still_too_small"] == []  # too-few groups are not double-reported


def test_run_missing_input_raises(tmp_path):
    cfg = _config(tmp_path)
    cfg["mitigation"]["input_path"] = str(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        run(cfg)


def test_run_unknown_method_raises(tmp_path):
    with pytest.raises(ValueError):
        run(_config(tmp_path, method="magic"))