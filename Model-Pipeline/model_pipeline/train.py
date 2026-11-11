"""Stage train: hyperparameter search for every model family.

Each family in params.yaml (models) is tried with its search space; every
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
    columns = data.load_columns(ctx)
    train, val = data.load_split(ctx, "train"), data.load_split(ctx, "val")
    X, y = train[columns].to_numpy(dtype=float), train["label"].to_numpy()
    Xv, yv = val[columns].to_numpy(dtype=float), val["label"].to_numpy()

    trials, candidates = [], []
    for family, spec in ctx.params["models"].items():
        if family not in FAMILIES:
            logger.warning("no model code for %s; skipped", family)
            continue
        best = None
        for number, hp in enumerate(param_grid(spec, seed)):
            model = fit(FAMILIES[family](hp, y, seed), X, y)
            m = metrics.compute(yv, predict(model, Xv), ev["threshold"],
                                ev["top_k_share"])
            trials.append({"family": family, "trial": number, **hp,
                           **{f"val_{k}": v for k, v in m.items()}})
            tracking.log_trial(ctx, family, number, hp, m)
            logger.info("%s trial %d %s: val PR-AUC %.4f", family, number,
                        hp, m["pr_auc"])
            if best is None or m["pr_auc"] > best[2]["pr_auc"]:
                best = (hp, model, m)

        hp, model, m = best
        joblib.dump(model, ctx.path("models", f"{family}.joblib"))
        candidates.append({"name": family, "family": family, "params": hp,
                           "mitigation": "none",
                           "model_file": f"models/{family}.joblib"})
        if train["weight"].gt(1.01).any():
            weighted = fit(FAMILIES[family](hp, y, seed), X, y,
                           train["weight"].to_numpy(dtype=float))
            joblib.dump(weighted,
                        ctx.path("models", f"{family}_reweighted.joblib"))
            candidates.append({
                "name": f"{family}+reweighted", "family": family,
                "params": hp, "mitigation": "reweighted",
                "model_file": f"models/{family}_reweighted.joblib"})
        tracking.log_params(ctx, {f"best_{family}_{k}": v
                                  for k, v in hp.items()})

    pd.DataFrame(trials).to_csv(ctx.path("reports", "tuning_results.csv"),
                                index=False)
    ctx.write_json("candidates.json", candidates)
    logger.info("candidates: %s", [c["name"] for c in candidates])
    return candidates


def load_model(ctx, candidate):
    return joblib.load(ctx.path(candidate["model_file"]))
