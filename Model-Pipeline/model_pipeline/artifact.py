"""The packaged model as the test stage writes it (artifact/ in the work
folder, and the same files in the bucket for every registry version):

    model.joblib      the trained model, any object with predict_proba(X)
    model_card.json   feature columns, threshold, the candidate (with its
                      per-group cuts if it is a group_thresholds model),
                      metrics and the data version

Used by the serving container and by the register stage (to score the
current champion on the new test set). Loading is generic: joblib/pickle
of any object with predict_proba, so a PyTorch model wrapped in such a
class loads the same way as the sklearn and XGBoost ones.
"""

import json
from pathlib import Path

import numpy as np

CARD = "model_card.json"
MODEL_FILES = ("model.joblib", "model.pkl", "model.pt")


def shift_scores(scores, groups, cuts, threshold):
    """Per-group decision thresholds, as in the data pipeline's
    scripts/bias_model.shift_scores (copied so that serving does not need
    the data pipeline): the score is moved so that score >= threshold
    exactly when it is at or above its group's cut."""
    eps = 1e-6
    p = np.clip(np.asarray(scores, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    cut = np.array([cuts.get(g, threshold) for g in groups], dtype=float)
    cut = np.clip(cut, eps, 1 - eps)
    base = np.log(threshold / (1 - threshold))
    return 1 / (1 + np.exp(-(logit - np.log(cut / (1 - cut)) + base)))


def _load_model(path):
    if path.suffix in (".pt", ".pth"):
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    import joblib
    return joblib.load(path)


class PackagedModel:
    """A model artifact folder, ready to score rows."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.card = json.loads((self.folder / CARD).read_text())
        name = self.card.get("model_file")
        candidates = [name] if name else MODEL_FILES
        path = next((self.folder / n for n in candidates
                     if (self.folder / n).exists()), None)
        if path is None:
            raise FileNotFoundError(f"no model file in {self.folder}")
        self.model = _load_model(path)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"{path.name} has no predict_proba")
        cand = self.card.get("candidate", {})
        self.name = cand.get("name", "model")
        self.columns = list(self.card["feature_columns"])
        self.threshold = float(self.card.get("threshold", 0.5))
        self.cuts = self.card.get("cuts", cand.get("cuts")) or {}
        self.group_slice = self.card.get("threshold_slice",
                                         cand.get("threshold_slice"))

    def missing_columns(self, available):
        return [c for c in self.columns if c not in set(available)]

    def probabilities(self, X):
        return np.asarray(self.model.predict_proba(
            np.asarray(X, dtype=float))[:, 1], dtype=float)

    def scores(self, X, groups=None):
        """Scores with the group thresholds applied (what the pipeline's
        metrics use); score >= threshold means "will stop"."""
        p = self.probabilities(X)
        if self.cuts and groups is not None:
            return shift_scores(p, groups, self.cuts, self.threshold)
        return p

    def group_threshold(self, group):
        return float(self.cuts.get(group, self.threshold)) \
            if self.cuts else self.threshold

    def score_frame(self, rows):
        """Scores of a pandas DataFrame (a split of the pipeline)."""
        groups = None
        if self.cuts and self.group_slice in rows.columns:
            groups = rows[self.group_slice].astype(str).to_numpy()
        return self.scores(rows[self.columns].to_numpy(dtype=float), groups)


def load(folder):
    return PackagedModel(folder)
