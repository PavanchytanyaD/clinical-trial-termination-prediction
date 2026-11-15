"""The packaged model as the test stage writes it (artifact/ in the work
folder, and the same files in the bucket for every registry version):

    model.joblib       the trained model, any object with predict_proba(X)
    model_card.json    feature columns, threshold, the candidate (with its
                       per-group thresholds if it is a group_thresholds
                       model), calibration, metrics and the data version
    calibrator.joblib  raw score -> probability of stopping (optional)
    text_pca.joblib    BioBERT vectors -> the model's txt_* features, for a
                       model with text features (optional)

Used by the serving container and by the register stage (to score the
current champion on the new test set). Loading is generic: joblib/pickle
of any object with predict_proba, so the scikit-learn, XGBoost and neural
network models load the same way.
"""

import json
from pathlib import Path

import numpy as np

CARD = "model_card.json"
MODEL_FILES = ("model.joblib", "model.pkl", "model.pt")
CALIBRATOR = "calibrator.joblib"
TEXT_PCA = "text_pca.joblib"


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


def threshold_steps(cand, card=None):
    """The per-group threshold steps of a candidate (one slice after the
    other); older cards keep one slice in cuts / threshold_slice."""
    card = card or {}
    if cand.get("threshold_steps"):
        return [{"slice": s["slice"], "cuts": s["cuts"] or {}}
                for s in cand["threshold_steps"]]
    cuts = card.get("cuts", cand.get("cuts"))
    slice_ = card.get("threshold_slice", cand.get("threshold_slice"))
    return [{"slice": slice_, "cuts": cuts}] if cuts and slice_ else []


def _load_model(path):
    if path.suffix in (".pt", ".pth"):
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    import joblib
    return joblib.load(path)


class PackagedModel:
    """A model artifact folder, ready to score rows."""

    def __init__(self, folder):
        import joblib

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
        self.steps = threshold_steps(cand, self.card)
        # one-slice models keep the simple interface (cuts, group_slice)
        self.cuts = self.steps[0]["cuts"] if len(self.steps) == 1 else {}
        self.group_slice = self.steps[0]["slice"] if self.steps else None
        self.group_slices = [s["slice"] for s in self.steps]
        cal = self.folder / CALIBRATOR
        self.calibrator = joblib.load(cal) if cal.exists() else None
        pca = self.folder / TEXT_PCA
        self.text = joblib.load(pca) if pca.exists() else None

    @property
    def has_groups(self):
        return bool(self.steps)

    def text_values(self, embeddings):
        """The txt_* features of one trial from its BioBERT vectors
        ({field: [768 numbers]}); a field without a vector gets the zero
        vector, as in training."""
        if self.text is None:
            raise ValueError("this model has no text features")
        out = {}
        for field in self.text["fields"]:
            pca = self.text["reducers"][field]
            vec = embeddings.get(field)
            vec = np.zeros(pca.n_features_in_) if vec is None \
                else np.asarray(vec, dtype=float)
            if vec.shape != (pca.n_features_in_,):
                raise ValueError(f"{field}: {vec.size} numbers, expected "
                                 f"{pca.n_features_in_}")
            comps = pca.transform(vec.reshape(1, -1))[0]
            prefix = f"txt_{field}_"
            names = [c for c in self.text["columns"] if c.startswith(prefix)]
            out.update(zip(names, comps))
        return out

    def missing_columns(self, available):
        return [c for c in self.columns if c not in set(available)]

    def probabilities(self, X):
        return np.asarray(self.model.predict_proba(
            np.asarray(X, dtype=float))[:, 1], dtype=float)

    def calibrated(self, scores):
        """Probability of stopping for raw model scores (the calibrate
        stage's calibrator), or None without one."""
        if self.calibrator is None:
            return None
        s = np.asarray(scores, dtype=float)
        if hasattr(self.calibrator, "predict_proba"):
            return self.calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.asarray(self.calibrator.predict(s), dtype=float)

    def scores(self, X, groups=None):
        """Scores with the group thresholds applied (what the pipeline's
        metrics use); score >= threshold means "will stop". groups: one
        array of group names per threshold step's slice ({slice: array}),
        or one array for a one-slice model."""
        s = self.probabilities(X)
        if not self.steps or groups is None:
            return s
        if not isinstance(groups, dict):
            groups = {self.group_slice: groups}
        for step in self.steps:
            g = groups.get(step["slice"])
            if g is not None:
                s = shift_scores(s, g, step["cuts"], self.threshold)
        return s

    def group_threshold(self, group):
        return float(self.cuts.get(group, self.threshold)) \
            if self.cuts else self.threshold

    def score_frame(self, rows):
        """Scores of a pandas DataFrame (a split of the pipeline)."""
        groups = {s: rows[s].astype(str).to_numpy()
                  for s in self.group_slices if s in rows.columns}
        return self.scores(rows[self.columns].to_numpy(dtype=float),
                           groups or None)


def load(folder):
    return PackagedModel(folder)
