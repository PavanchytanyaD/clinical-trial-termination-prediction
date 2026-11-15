"""Tests for the shared tracking address and the run comparison."""

import os
from types import SimpleNamespace

import pytest

from model_pipeline import tracking


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)


def make_ctx(tmp_path, uri=None):
    settings = {"experiment": "test-exp"}
    if uri is not None:
        settings["uri"] = uri
    return SimpleNamespace(params={"tracking": settings}, work=tmp_path)


# --- tracking_uri -------------------------------------------------------------

def test_env_wins_over_params(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    assert tracking.tracking_uri(make_ctx(tmp_path, "http://params:5000")) \
        == "http://env:5000"


def test_params_used_when_no_env(tmp_path):
    assert tracking.tracking_uri(make_ctx(tmp_path, "http://shared:5000")) \
        == "http://shared:5000"
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ  # server, no flag


def test_falls_back_to_work_folder(tmp_path):
    uri = tracking.tracking_uri(make_ctx(tmp_path))
    assert uri == (tmp_path / "mlruns").resolve().as_uri()
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_shared_folder_in_params_sets_file_store_flag(tmp_path):
    shared = str(tmp_path / "team_mlruns")
    assert tracking.tracking_uri(make_ctx(tmp_path, shared)) == shared
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_empty_uri_in_params_falls_back(tmp_path):
    uri = tracking.tracking_uri(make_ctx(tmp_path, ""))
    assert uri.endswith("/mlruns")


def test_password_hidden_in_log_line():
    assert tracking._safe("https://user:secret@host:5000/x") \
        == "https://host:5000/x"
    assert tracking._safe("http://host:5000") == "http://host:5000"


# --- compare_runs ------------------------------------------------------------

def _log_pipeline_run(folder, name, pr_auc, model):
    """Write one pipeline run plus one nested trial into an mlruns folder."""
    import mlflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(folder.resolve().as_uri())
    mlflow.set_experiment("test-exp")
    with mlflow.start_run(run_name=name):
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_param("seed", 42)
        mlflow.set_tag("selected_model", model)
        with mlflow.start_run(run_name="xgboost-01", nested=True):
            mlflow.log_metric("val_pr_auc", 0.1)


def test_collect_runs_combines_folders_and_skips_trials(tmp_path):
    pytest.importorskip("mlflow")
    from model_pipeline import compare_runs

    a, b = tmp_path / "a" / "mlruns", tmp_path / "b" / "mlruns"
    _log_pipeline_run(a, "run-a", 0.40, "xgboost")
    _log_pipeline_run(b, "run-b", 0.35, "logreg")

    runs = compare_runs.collect_runs([str(a), str(b)], "test-exp")
    assert sorted(runs["run_name"]) == ["run-a", "run-b"]  # no nested trials
    assert set(runs["selected_model"]) == {"xgboost", "logreg"}
    assert runs["param.seed"].tolist() == ["42", "42"]


def test_main_writes_table_and_chart(tmp_path):
    pytest.importorskip("mlflow")
    from model_pipeline import compare_runs

    folder = tmp_path / "mlruns"
    _log_pipeline_run(folder, "run-a", 0.40, "xgboost")
    out = tmp_path / "report"
    code = compare_runs.main(["--uri", str(folder), "--experiment",
                              "test-exp", "--out", str(out)])
    assert code == 0
    assert (out / "run_comparison.csv").exists()
    assert (out / "run_comparison.png").exists()


def test_missing_experiment_gives_no_rows(tmp_path):
    pytest.importorskip("mlflow")
    from model_pipeline import compare_runs

    folder = tmp_path / "mlruns"
    _log_pipeline_run(folder, "run-a", 0.40, "xgboost")
    assert compare_runs.collect_runs([str(folder)], "nope").empty


def test_missing_folder_fails_with_exit_code_1(tmp_path):
    pytest.importorskip("mlflow")
    from model_pipeline import compare_runs

    assert compare_runs.main(["--uri", str(tmp_path / "missing")]) == 1


def test_pick_metric_falls_back_to_a_pr_auc_column():
    import pandas as pd
    from model_pipeline import compare_runs

    runs = pd.DataFrame({"val_logreg_pr_auc": [0.3], "brier": [0.1]})
    assert compare_runs.pick_metric(runs, "test_pr_auc") == \
        "val_logreg_pr_auc"
    with pytest.raises(KeyError):
        compare_runs.pick_metric(pd.DataFrame({"brier": [0.1]}))
