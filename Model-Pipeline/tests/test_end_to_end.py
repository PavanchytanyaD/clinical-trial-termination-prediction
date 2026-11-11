"""Runs the stages load_data to sensitivity on a small synthetic data set
in the data pipeline's formats (no cloud access, no registry)."""

import json

import numpy as np
import pandas as pd

from model_pipeline import run as runner
from model_pipeline.config import load_params


def make_data(base, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    p = 1 / (1 + np.exp(-(X[:, 0] * 1.5 - X[:, 1] - 1.2)))
    label = (rng.random(n) < p).astype(float)
    label[rng.random(n) < 0.1] = np.nan          # still running trials
    ids = [f"NCT{i:08d}" for i in range(n)]
    dates = pd.date_range("2015-01-01", periods=n, freq="D").astype(str)
    feats = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    feats.insert(0, "nct_id", ids)
    feats["label"] = label
    feats["study_first_submitted_date"] = dates
    proc = base / "data" / "processed"
    proc.mkdir(parents=True)
    feats.to_csv(proc / "structured_features.csv", index=False)
    (proc / "structured_features.meta.json").write_text(json.dumps(
        {"feature_columns": [f"f{i}" for i in range(6)],
         "leakage_risk": []}))
    regions = rng.choice(["US_only", "non_US", "mixed"], n)
    phases = rng.choice(["PHASE1", "PHASE2", "PHASE3"], n)
    pd.DataFrame({"nct_id": ids, "region": regions, "phase": phases}) \
        .to_csv(proc / "trial_slices.csv", index=False)


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    make_data(tmp_path / "dp")
    params = load_params()
    params["data"].update(source="local", local_dir=str(tmp_path / "dp"))
    params["models"] = {"logreg": {"search": "grid",
                                   "space": {"C": [0.1, 1.0]}}}
    params["bias"].update(slices=["region", "phase"], min_group_size=20,
                          min_stopped=3)
    params["gate"].update(min_pr_auc=0.05, min_roc_auc=0.5,
                          block_bias_gap=1.0)
    monkeypatch.setattr(runner, "load_params", lambda: params)
    monkeypatch.setattr("model_pipeline.notify.alert",
                        lambda *a, **k: None)
    work = tmp_path / "work"
    runner.main(["--stage", "all", "--workdir", str(work),
                 "--skip-register"])

    selection = json.loads((work / "reports/selection.json").read_text())
    assert selection["selected"]["eligible"]
    names = {c["name"] for c in selection["candidates"]}
    assert {"logreg", "logreg+group_thresholds"} <= names
    card = json.loads((work / "artifact/model_card.json").read_text())
    assert card["test_metrics"]["pr_auc"] > 0
    assert (work / "artifact/model.joblib").exists()
    assert (work / "reports/bias/bias_report.md").exists()
    assert (work / "reports/tuning_results.csv").exists()
