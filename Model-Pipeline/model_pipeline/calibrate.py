"""Stage calibrate: map the selected model's scores to probabilities that
match how often trials really stop.

The models are trained with the classes balanced, so their scores run far
above the real stop rate (a score of 0.6 does not mean a 60% chance).
Two calibrators are tried on the validation set: sigmoid (Platt scaling, a
logistic curve on the score) and isotonic (a step function that only goes
up). Each is scored by its 5-fold cross-validated Brier score on the
validation set and the better one is refitted on all of it. The test stage
reports it on the test set and packages it with the model.

Decisions do not change: a trial is flagged when the model's own score
reaches the threshold, as in every gate and bias check. The calibrated
probability is the risk shown to a reviewer.
Outputs: models/calibrator.joblib, reports/calibration/.
"""

import logging

import joblib
import numpy as np

from model_pipeline import data, metrics, plots, tracking
from model_pipeline.models import score_candidate
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)
METHODS = ("sigmoid", "isotonic")


def fit_calibrator(method, s, y):
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1e6).fit(np.reshape(s, (-1, 1)), y)
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip") \
        .fit(s, y)


def apply(calibrator, s):
    """Calibrated probabilities for raw scores."""
    s = np.asarray(s, dtype=float)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
    return calibrator.predict(s)


def cv_brier(method, s, y, seed, folds=5):
    """Out-of-fold Brier score of a calibrator fitted on the other folds."""
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros(len(y))
    split = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fit_idx, held in split.split(s.reshape(-1, 1), y):
        out[held] = apply(fit_calibrator(method, s[fit_idx], y[fit_idx]),
                          s[held])
    return float(np.mean((out - y) ** 2))


def selected_candidate(ctx):
    name = ctx.read_json("reports/selection.json")["selected"]["name"]
    return next(c for c in ctx.read_json("candidates.json")
                if c["name"] == name)


def run(ctx):
    ev, seed = ctx.params["evaluation"], int(ctx.params["seed"])
    cand = selected_candidate(ctx)
    val = data.load_split(ctx, "val")
    s = score_candidate(cand, load_model(ctx, cand), val,
                        data.load_columns(ctx, cand), ev["threshold"])
    y = val["label"].to_numpy().astype(int)

    scores = {m: cv_brier(m, s, y, seed) for m in METHODS}
    method = min(scores, key=scores.get)
    calibrator = fit_calibrator(method, s, y)
    joblib.dump(calibrator, ctx.path("models", "calibrator.joblib"))

    p = apply(calibrator, s)
    result = {"candidate": cand["name"], "method": method,
              "cv_brier": scores,
              "val_brier_raw": float(np.mean((np.clip(s, 0, 1) - y) ** 2)),
              "val_ece_raw": metrics.expected_calibration_error(y, s),
              "val_ece_calibrated": metrics.expected_calibration_error(y, p),
              "val_mean_score_raw": float(np.mean(s)),
              "val_mean_probability": float(np.mean(p)),
              "val_stopped_rate": float(np.mean(y))}
    out = ctx.path("reports", "calibration", ".keep").parent
    ctx.write_json("reports/calibration/calibration.json", result)
    plots.reliability({"raw score": (y, s), f"calibrated ({method})":
                       (y, p)}, out / "reliability_val.png",
                      f"Reliability, validation set ({cand['name']})")
    tracking.log_metrics(ctx, {k: v for k, v in result.items()
                               if isinstance(v, float)}, prefix="cal_")
    tracking.log_params(ctx, {"calibration_method": method})
    tracking.log_artifacts(ctx, out, "calibration")
    logger.info("calibration: %s (CV Brier %s); mean score %.3f -> %.3f, "
                "stop rate %.3f", method,
                {k: round(v, 4) for k, v in scores.items()},
                result["val_mean_score_raw"], result["val_mean_probability"],
                result["val_stopped_rate"])
    return result
