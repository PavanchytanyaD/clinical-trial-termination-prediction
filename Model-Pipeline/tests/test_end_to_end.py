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
    return ids, X, label, dates


def add_text_and_stops(base, ids, X, label, dates, seed=0):
    """Embeddings in the data pipeline's format (signal in one field) and
    an AACT studies table with stop dates."""
    rng = np.random.default_rng(seed)
    emb = base / "data" / "processed" / "embeddings"
    emb.mkdir(parents=True)
    (emb / "ids.txt").write_text("\n".join(ids[:-50]) + "\n")  # some miss
    for field in ("title", "inclusion"):
        vecs = rng.normal(size=(len(ids) - 50, 32))
        if field == "inclusion":
            vecs[:, 0] += 2 * X[:-50, 2]
        np.save(emb / f"{field}.npy", vecs.astype(np.float16))
    (emb / "meta.json").write_text("{}")
    start = pd.to_datetime(pd.Series(dates))
    stop = start + pd.to_timedelta(rng.integers(30, 1500, len(ids)), "D")
    raw = base / "data" / "raw" / "aact_snapshot"
    raw.mkdir(parents=True)
    pd.DataFrame({
        "nct_id": ids,
        "completion_date": stop.dt.strftime("%Y-%m-%d"),
        "completion_date_type": np.where(label == 1, "ACTUAL", "ESTIMATED"),
        "primary_completion_date": "",
        "primary_completion_date_type": "",
        "last_update_submitted_date": stop.dt.strftime("%Y-%m-%d"),
    }).to_csv(raw / "studies.txt", sep="|", index=False)


def run_pipeline(tmp_path, monkeypatch, models):
    params = load_params()
    params["data"].update(source="local", local_dir=str(tmp_path / "dp"))
    params["data"]["embeddings"].update(fields=["title", "inclusion"],
                                        pca_components=4)
    params["models"] = models
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
    return work


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    make_data(tmp_path / "dp")
    work = run_pipeline(tmp_path, monkeypatch, {
        "logreg": {"search": "grid", "space": {"C": [0.1, 1.0]}}})

    selection = json.loads((work / "reports/selection.json").read_text())
    assert selection["selected"]["eligible"]
    names = {c["name"] for c in selection["candidates"]}
    assert {"logreg", "logreg+group_thresholds"} <= names
    card = json.loads((work / "artifact/model_card.json").read_text())
    assert card["test_metrics"]["pr_auc"] > 0
    assert (work / "artifact/model.joblib").exists()
    assert (work / "reports/bias/bias_report.md").exists()
    assert (work / "reports/tuning_results.csv").exists()
    assert (work / "artifact/calibrator.joblib").exists()
    assert "No stop dates" in (work / "reports/lead_time/lead_time.md") \
        .read_text()


def test_pipeline_with_text_features_nn_and_lead_time(tmp_path,
                                                      monkeypatch):
    ids, X, label, dates = make_data(tmp_path / "dp")
    add_text_and_stops(tmp_path / "dp", ids, X, label, dates)
    work = run_pipeline(tmp_path, monkeypatch, {
        "logreg": {"search": "grid", "space": {"C": [1.0]}},
        "nn": {"text_features": True, "search": "grid",
               "space": {"hidden_layer_sizes": [[16]], "alpha": [0.001]}}})

    text_cols = json.loads((work / "data/text_columns.json").read_text())
    assert len(text_cols) == 8                    # 2 fields x 4 components
    candidates = json.loads((work / "candidates.json").read_text())
    by_name = {c["name"]: c for c in candidates}
    assert by_name["nn"]["text_features"] is True
    assert by_name["logreg"]["text_features"] is False
    assert "nn+group_thresholds" in by_name

    card = json.loads((work / "artifact/model_card.json").read_text())
    assert card["calibration"]["method"] in ("sigmoid", "isotonic")
    assert "brier_calibrated" in card["test_metrics"]
    if card["candidate"]["text_features"]:
        assert (work / "artifact/text_pca.joblib").exists()
        assert set(text_cols) <= set(card["feature_columns"])

    lead = json.loads((work / "reports/lead_time/lead_time.json")
                      .read_text())
    assert lead["true_positives"]["n"] > 0
    assert lead["true_positives"]["median_months"] > 0
    assert (work / "reports/lead_time/top_false_negatives.csv").exists()
    test = pd.read_pickle(work / "data/test.pkl")
    assert "stop_date" in test and "stop_date" not in card["feature_columns"]
