"""Unit tests for the model pipeline's decision logic (no cloud access)."""

import numpy as np
import pytest

from model_pipeline import data, metrics, models, registry, select

LOCK = {"stages": {
    "preprocess": {"outs": [
        {"path": "data/processed/structured_features.csv", "md5": "ab12"},
        {"path": "data/processed/structured_features.meta.json",
         "md5": "cd34"}]},
    "bias": {"outs": [{"path": "reports/bias", "md5": "ef56.dir"}]}}}

GATE = {"min_pr_auc": 0.3, "min_roc_auc": 0.6, "alert_bias_gap": 0.1,
        "block_bias_gap": 0.25, "bias_penalty": 0.5, "min_improvement": 0}


def test_find_out_file_and_directory():
    assert data.find_out(LOCK, "data/processed/structured_features.csv") \
        == ("ab12", None)
    assert data.find_out(LOCK, "reports/bias/mitigation/trial_weights.csv") \
        == ("ef56.dir", "mitigation/trial_weights.csv")
    assert data.find_out(LOCK, "data/other.csv") == (None, None)


def test_remote_paths_cover_dvc2_and_dvc3():
    paths = data.remote_paths("b/dvc-store", "abcdef")
    assert paths == ["b/dvc-store/files/md5/ab/cdef", "b/dvc-store/ab/cdef"]


def test_param_grid_grid_and_random():
    grid = models.param_grid({"search": "grid",
                              "space": {"a": [1, 2], "b": [3, 4, 5]}}, 0)
    assert len(grid) == 6
    spec = {"search": "random", "n_iter": 4,
            "space": {"a": [1, 2, 3], "b": [4, 5, 6]}}
    first, again = models.param_grid(spec, 7), models.param_grid(spec, 7)
    assert len(first) == 4 and first == again
    assert len({tuple(sorted(d.items())) for d in first}) == 4


def test_choose_prefers_fair_model_and_blocks_biased():
    rows = {"a": {"pr_auc": 0.50, "roc_auc": 0.8, "max_gap": 0.30},
            "b": {"pr_auc": 0.46, "roc_auc": 0.8, "max_gap": 0.05},
            "c": {"pr_auc": 0.48, "roc_auc": 0.8, "max_gap": 0.20}}
    winner, table = select.choose(rows, GATE)
    assert winner["name"] == "b"                  # 0.46-0.025 > 0.48-0.10
    assert not next(t for t in table if t["name"] == "a")["eligible"]


def test_choose_returns_none_when_all_fail():
    rows = {"a": {"pr_auc": 0.2, "roc_auc": 0.8, "max_gap": 0.0}}
    winner, _ = select.choose(rows, GATE)
    assert winner is None


@pytest.mark.parametrize("new,old,expected", [
    (0.40, None, True), (0.41, 0.40, True), (0.40, 0.40, True),
    (0.39, 0.40, False)])
def test_should_promote(new, old, expected):
    assert registry.should_promote(new, old, 0.0) is expected


def test_label_is_valid_for_vertex():
    assert registry.label("XGBoost+Group_Thresholds") == \
        "xgboost-group_thresholds"
    assert len(registry.label("x" * 100)) == 63


def test_precision_at_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .2])
    assert metrics.precision_at_k(y, s, 0.3) == pytest.approx(2 / 3)
