"""Stage sensitivity: which features and hyperparameters matter.

1. Feature importance with SHAP on the selected model (a sample of
   validation trials): reports/sensitivity/shap_importance.csv and a chart
   of the 20 features with the largest mean |SHAP value|; for a model with
   text features also the importance per text field (shap_text_fields).
2. Hyperparameter sensitivity from the train stage's search
   (tuning_results.csv): mean validation PR-AUC for each value of each
   hyperparameter, and its spread (max - min), which ranks how much a
   hyperparameter changes the result.
"""

import logging

import pandas as pd

from model_pipeline import data, plots, tracking
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


def shap_importance(model, family, X, columns, seed, background=100):
    import shap
    if family == "xgboost":
        values = shap.TreeExplainer(model).shap_values(X)
    else:
        bg = shap.sample(X, min(background, len(X)), random_state=seed)
        explainer = shap.Explainer(lambda a: model.predict_proba(a)[:, 1],
                                   shap.maskers.Independent(bg))
        # the permutation explainer needs at least 2 * features + 1 calls
        values = explainer(X, max_evals=2 * X.shape[1] + 1).values
    return pd.Series(abs(values).mean(axis=0), index=columns) \
        .sort_values(ascending=False)


def text_field_importance(imp):
    """Mean |SHAP| summed over each text field's PCA columns
    (txt_<field>_<i>) beside the structured features' total."""
    text = imp[imp.index.str.startswith("txt_")]
    if text.empty:
        return pd.Series(dtype=float)
    fields = text.index.str.replace(r"^txt_(.+)_\d+$", r"\1", regex=True)
    out = text.groupby(fields).sum()
    out["(all structured features)"] = imp.drop(text.index).sum()
    return out.sort_values(ascending=False)


def hyperparameter_sensitivity(tuning):
    rows = []
    skip = {"family", "trial"}
    for family, g in tuning.groupby("family"):
        for col in g.columns:
            if col in skip or col.startswith("val_") or \
                    g[col].nunique() < 2:
                continue
            means = g.groupby(col)["val_pr_auc"].mean()
            rows.append({"family": family, "hyperparameter": col,
                         "spread": float(means.max() - means.min()),
                         **{f"pr_auc@{k}": float(v)
                            for k, v in means.items()}})
    return pd.DataFrame(rows).sort_values("spread", ascending=False) \
        if rows else pd.DataFrame()


def run(ctx):
    out = ctx.path("reports", "sensitivity", ".keep").parent
    selected = ctx.read_json("reports/selection.json")["selected"]
    cand = next(c for c in ctx.read_json("candidates.json")
                if c["name"] == selected["name"])
    columns = data.load_columns(ctx, cand)
    val = data.load_split(ctx, "val")
    sample = val.sample(min(300, len(val)), random_state=ctx.params["seed"])
    try:
        imp = shap_importance(load_model(ctx, cand), cand["family"],
                              sample[columns].to_numpy(dtype=float),
                              columns, ctx.params["seed"])
        imp.rename("mean_abs_shap").to_csv(out / "shap_importance.csv")
        plots.bars(imp.head(20), out / "shap_top20.png",
                   f"Top 20 features by mean |SHAP| ({cand['name']})",
                   "mean |SHAP value|")
        fields = text_field_importance(imp)
        if not fields.empty:
            fields.rename("mean_abs_shap").to_csv(
                out / "shap_text_fields.csv")
            plots.bars(fields, out / "shap_text_fields.png",
                       f"Mean |SHAP| per text field ({cand['name']})",
                       "sum of mean |SHAP value| over the field's columns")
    except Exception as exc:
        logger.warning("SHAP skipped: %s", exc)

    tuning = pd.read_csv(ctx.path("reports", "tuning_results.csv"))
    sens = hyperparameter_sensitivity(tuning)
    if not sens.empty:
        sens.to_csv(out / "hyperparameter_sensitivity.csv", index=False)
        for family, g in sens.groupby("family"):
            plots.bars(g.set_index("hyperparameter")["spread"],
                       out / f"hyperparameters_{family}.png",
                       f"{family}: PR-AUC spread per hyperparameter",
                       "max - min mean validation PR-AUC")
    tracking.log_artifacts(ctx, out, "sensitivity")
