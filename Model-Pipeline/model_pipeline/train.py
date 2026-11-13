"""Stage train: hyperparameter search for every model family.

Each entry in params.yaml (models) is tried with its search space, on the
structured features or, with text_features: true, on the structured plus
the BioBERT text features; every
trial is trained on the training set, scored on the validation set and
logged to MLflow as a nested run. The best trial per family (by validation
PR-AUC) is kept, and trained a second time with the bias stage's trial
weights (under-represented groups weighted up) as a mitigation candidate.
"""

import logging

import joblib
import pandas as pd

from model_pipeline import data, metrics, tracking
from model_pipeline.models import FAMILIES, fit, param_grid, predict

logger = logging.getLogger(__name__)


def run(ctx):
    seed = int(ctx.params["seed"])
    ev = ctx.params["evaluation"]
    train, val = data.load_split(ctx, "train"), data.load_split(ctx, "val")
    y, yv = train["label"].to_numpy(), val["label"].to_numpy()
    has_text = bool(data.load_text_columns(ctx))

    trials, candidates = [], []
    for name, spec in ctx.params["models"].items():
        family = spec.get("family", name)
        if family not in FAMILIES:
            logger.warning("no model code for %s; skipped", family)
            continue
        text = bool(spec.get("text_features")) and has_text
        if spec.get("text_features") and not has_text:
            logger.warning("%s: no text features in this data version",
                           name)
        columns = data.load_columns(ctx, {"text_features": text})
        X = train[columns].to_numpy(dtype=float)
        Xv = val[columns].to_numpy(dtype=float)
        best = None
        for number, hp in enumerate(param_grid(spec, seed)):
            model = fit(FAMILIES[family](hp, y, seed), X, y)
            m = metrics.compute(yv, predict(model, Xv), ev["threshold"],
                                ev["top_k_share"])
            trials.append({"family": name, "trial": number, **hp,
                           **{f"val_{k}": v for k, v in m.items()}})
            tracking.log_trial(ctx, name, number, hp, m)
            logger.info("%s trial %d %s: val PR-AUC %.4f", name, number,
                        hp, m["pr_auc"])
            if best is None or m["pr_auc"] > best[2]["pr_auc"]:
                best = (hp, model, m)

        hp, model, m = best
        joblib.dump(model, ctx.path("models", f"{name}.joblib"))
        candidates.append({"name": name, "family": family, "params": hp,
                           "text_features": text, "mitigation": "none",
                           "model_file": f"models/{name}.joblib"})
        if train["weight"].gt(1.01).any():
            weighted = fit(FAMILIES[family](hp, y, seed), X, y,
                           train["weight"].to_numpy(dtype=float))
            joblib.dump(weighted,
                        ctx.path("models", f"{name}_reweighted.joblib"))
            candidates.append({
                "name": f"{name}+reweighted", "family": family,
                "params": hp, "text_features": text,
                "mitigation": "reweighted",
                "model_file": f"models/{name}_reweighted.joblib"})
        tracking.log_params(ctx, {f"best_{name}_{k}": v
                                  for k, v in hp.items()})

    pd.DataFrame(trials).to_csv(ctx.path("reports", "tuning_results.csv"),
                                index=False)
    ctx.write_json("candidates.json", candidates)
    logger.info("candidates: %s", [c["name"] for c in candidates])
    return candidates


def load_model(ctx, candidate):
    return joblib.load(ctx.path(candidate["model_file"]))
