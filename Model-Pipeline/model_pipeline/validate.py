"""Stage validate: every candidate on the hold-out validation set (never
used for training): PR-AUC, ROC-AUC, Brier, precision, recall, F1,
Precision@K, plus comparison charts."""

import logging

from model_pipeline import data, metrics, plots, tracking
from model_pipeline.models import score_candidate
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


def run(ctx):
    ev = ctx.params["evaluation"]
    columns = data.load_columns(ctx)
    val = data.load_split(ctx, "val")
    results, curves = {}, {}
    for cand in ctx.read_json("candidates.json"):
        s = score_candidate(cand, load_model(ctx, cand), val, columns,
                            ev["threshold"])
        results[cand["name"]] = metrics.compute(
            val["label"], s, ev["threshold"], ev["top_k_share"])
        curves[cand["name"]] = (val["label"].to_numpy(), s)
        tracking.log_metrics(ctx, results[cand["name"]],
                             prefix=f"val_{cand['name']}_")
    ctx.write_json("reports/val_metrics.json", results)
    out = ctx.path("reports", "figures", ".keep").parent
    plots.compare_models(results, out / "val_model_comparison.png")
    plots.pr_curves(curves, out / "val_pr_curves.png",
                    "Precision-recall, validation set")
    tracking.log_artifacts(ctx, out, "figures")
    for name, m in results.items():
        logger.info("%s: val PR-AUC %.4f ROC-AUC %.4f", name, m["pr_auc"],
                    m["roc_auc"])
    return results
