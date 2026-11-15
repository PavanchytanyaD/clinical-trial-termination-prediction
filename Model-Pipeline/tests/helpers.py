"""Small fitted models and artifact folders for the serving and registry
tests."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLUMNS = ["f0", "f1", "f2"]
CUTS = {"US_only": 0.3, "non_US": 0.7}


def make_rows(n=400, seed=0):
    """Rows like a pipeline split: features, label and a region slice."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(COLUMNS)))
    p = 1 / (1 + np.exp(-(2.0 * X[:, 0] - X[:, 1] - 0.5)))
    rows = pd.DataFrame(X, columns=COLUMNS)
    rows["label"] = (rng.random(n) < p).astype(int)
    rows["region"] = rng.choice(["US_only", "non_US", "mixed"], n)
    rows["nct_id"] = [f"NCT{i:08d}" for i in range(n)]
    return rows


def fit_logreg(rows, columns=COLUMNS):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    return model.fit(rows[columns].to_numpy(dtype=float), rows["label"])


def write_artifact(folder, model, name="logreg", cuts=None,
                   columns=COLUMNS, threshold=0.5, snapshot="20261027"):
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    cand = {"name": name, "family": name.split("+")[0],
            "mitigation": "group_thresholds" if cuts else "none",
            "model_file": f"models/{name}.joblib"}
    if cuts:
        cand.update(threshold_slice="region", cuts=cuts)
    card = {"candidate": cand, "feature_columns": list(columns),
            "threshold": threshold,
            "test_metrics": {"pr_auc": 0.5, "roc_auc": 0.7},
            "data_version": {"snapshot": snapshot}, "run_id": "run-test"}
    (folder / "model_card.json").write_text(json.dumps(card))
    return folder


class Constant:
    """A non-sklearn model with only predict_proba (like the PyTorch
    wrapper): every trial gets the same probability."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.column_stack([np.full(len(X), 1 - self.p),
                                np.full(len(X), self.p)])
