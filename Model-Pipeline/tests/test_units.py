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


def test_balanced_weights_equalize_classes():
    y = np.array([1, 0, 0, 0])
    w = models.balanced_weights(y)
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
    w2 = models.balanced_weights(y, np.array([2.0, 1, 1, 1]))
    assert w2[y == 1].sum() == pytest.approx(w2[y == 0].sum())


def test_nn_fits_and_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    y = (X[:, 0] + rng.normal(scale=0.5, size=3000) > 1).astype(int)
    hp = {"hidden_layer_sizes": [8], "alpha": 0.01, "n_seeds": 2}
    model = models.fit(models.build_nn(hp, y, 0), X, y)
    s = models.predict(model, X)
    assert s.shape == (3000,) and s[y == 1].mean() > s[y == 0].mean() + .3


def test_calibrator_moves_scores_to_the_stop_rate():
    from model_pipeline import calibrate
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, .15, 2000), 0, 1)
    for method in calibrate.METHODS:
        p = calibrate.apply(calibrate.fit_calibrator(method, s, y), s)
        assert abs(p.mean() - y.mean()) < 0.02
        assert np.mean((p - y) ** 2) < np.mean((s - y) ** 2)


def test_expected_calibration_error():
    y = np.array([0, 0, 1, 1] * 25)
    assert metrics.expected_calibration_error(y, np.full(100, .5)) == 0
    assert metrics.expected_calibration_error(y, np.full(100, .9)) \
        == pytest.approx(.4)


def test_stop_dates_prefer_actual_dates(tmp_path):
    import pandas as pd
    pd.DataFrame({
        "nct_id": ["A", "B", "C"],
        "completion_date": ["2021-05-01", "2022-01-01", ""],
        "completion_date_type": ["ACTUAL", "ESTIMATED", ""],
        "primary_completion_date": ["", "2021-03-01", ""],
        "primary_completion_date_type": ["", "ACTUAL", ""],
        "last_update_submitted_date": ["2023-01-01", "2023-01-01",
                                       "2020-02-02"],
    }).to_csv(tmp_path / "studies.txt", sep="|", index=False)
    s = data.stop_dates(tmp_path).set_index("nct_id")
    assert str(s.loc["A", "stop_date"].date()) == "2021-05-01"
    assert str(s.loc["B", "stop_date"].date()) == "2021-03-01"
    assert s.loc["C", "stop_date_source"] == "last_update"


def test_stop_dates_skip_a_table_without_date_columns(tmp_path):
    (tmp_path / "studies.txt").write_text(
        "nct_id|overall_status|start_date\nA|COMPLETED|2020-01-01\n")
    assert data.stop_dates(tmp_path) is None


class FakeBucket:
    """The few gcsfs calls data.download makes, over a dict of objects."""

    def __init__(self, objects):
        self.objects = objects

    def exists(self, path):
        return path in self.objects

    def get(self, path, dest):
        from pathlib import Path
        Path(dest).write_bytes(self.objects[path])

    def cat(self, path):
        return self.objects[path]


def test_download_from_lock_gets_embeddings_and_studies(tmp_path,
                                                        monkeypatch):
    import json as js
    import yaml
    remote = "b/dvc-store/files/md5"
    lock = {"stages": {
        "preprocess": {"outs": [
            {"path": "data/processed/structured_features.csv",
             "md5": "aa01"},
            {"path": "data/processed/structured_features.meta.json",
             "md5": "aa02"},
            {"path": "data/processed/trial_slices.csv", "md5": "aa03"}]},
        "embeddings": {"outs": [{"path": "data/processed/embeddings",
                                 "md5": "bb00.dir"}]},
        "acquire": {"outs": [{"path": "data/raw/aact_snapshot",
                              "md5": "cc00.dir"}]}}}
    objects = {
        "b/v/1/dvc.lock": yaml.safe_dump(lock).encode(),
        f"{remote}/aa/01": b"features", f"{remote}/aa/02": b"meta",
        f"{remote}/aa/03": b"slices",
        f"{remote}/bb/00.dir": js.dumps([
            {"relpath": "ids.txt", "md5": "bb01"},
            {"relpath": "meta.json", "md5": "bb02"},
            {"relpath": "title.npy", "md5": "bb03"}]).encode(),
        f"{remote}/bb/01": b"ids", f"{remote}/bb/02": b"{}",
        f"{remote}/bb/03": b"npy",
        f"{remote}/cc/00.dir": js.dumps([
            {"relpath": "studies.txt", "md5": "cc01"}]).encode(),
        f"{remote}/cc/01": b"studies"}
    import gcsfs
    monkeypatch.setattr(gcsfs, "GCSFileSystem",
                        lambda: FakeBucket(objects))

    class Ctx:
        params = {"data": {
            "source": "lock", "bucket": "b", "versions_prefix": "v",
            "dvc_remote": "dvc-store", "snapshot": "1",
            "files": {"features": "data/processed/structured_features.csv",
                      "meta": "data/processed/structured_features.meta.json",
                      "slices": "data/processed/trial_slices.csv",
                      "studies": "data/raw/aact_snapshot/studies.txt"},
            "embeddings": {"dir": "data/processed/embeddings",
                           "fields": ["title", "inclusion"]}}}

    version = data.download(Ctx(), tmp_path)
    assert (tmp_path / "studies.txt").read_bytes() == b"studies"
    assert (tmp_path / "embeddings" / "title.npy").read_bytes() == b"npy"
    assert (tmp_path / "embeddings" / "ids.txt").exists()
    assert not (tmp_path / "embeddings" / "inclusion.npy").exists()
    assert version["md5"]["embeddings/inclusion.npy"] is None


def test_lead_time_measures_only_actual_stop_dates():
    import pandas as pd
    from model_pipeline import lead_time
    rows = pd.DataFrame({
        "study_first_submitted_date": ["2019-01-01"] * 3,
        "stop_date": pd.to_datetime(["2020-01-01", "2021-01-01",
                                     "2024-01-01"]),
        "stop_date_source": ["completion", "primary_completion",
                             "last_update"]})
    months = lead_time.months_to_stop(rows)
    assert months.notna().tolist() == [True, True, False]


def test_lead_time_summary_leaves_out_after_the_fact():
    import pandas as pd
    from model_pipeline import lead_time
    out = lead_time.summary(pd.Series([-2.0, 6.0, 12.0, 30.0]))
    assert out["n"] == 4 and out["n_after_the_fact"] == 1
    assert out["median_months"] == 12.0
    assert out["share_over_12_months"] == pytest.approx(1 / 3)


def test_text_field_importance_groups_pca_columns():
    import pandas as pd
    from model_pipeline import sensitivity
    imp = pd.Series({"txt_inclusion_00": .3, "txt_inclusion_01": .1,
                     "txt_title_00": .05, "phase_2": .2, "n_arms": .1})
    out = sensitivity.text_field_importance(imp)
    assert out["inclusion"] == pytest.approx(.4)
    assert out["(all structured features)"] == pytest.approx(.3)


def test_precision_at_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .2])
    assert metrics.precision_at_k(y, s, 0.3) == pytest.approx(2 / 3)
