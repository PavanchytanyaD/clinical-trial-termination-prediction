"""Stage test: the selected model on the test set, the most recently
registered trials, which no earlier stage has looked at. The model must
pass the same gate here, then it is packaged for the registry:
artifact/model.joblib (the trained model) and artifact/model_card.json
(features, threshold, group thresholds, metrics and the data version)."""

import logging
import shutil

from model_pipeline import bias as bias_stage
from model_pipeline import data, metrics, notify, plots, tracking
from model_pipeline.models import score_candidate
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


def run(ctx):
    ev, gate, b = (ctx.params["evaluation"], ctx.params["gate"],
                   ctx.params["bias"])
    columns = data.load_columns(ctx)
    test = data.load_split(ctx, "test")
    selected = ctx.read_json("reports/selection.json")["selected"]
    cand = next(c for c in ctx.read_json("candidates.json")
                if c["name"] == selected["name"])
    model = load_model(ctx, cand)
    s = score_candidate(cand, model, test, columns, ev["threshold"])
    m = metrics.compute(test["label"], s, ev["threshold"], ev["top_k_share"])

    slice_names = [x for x in ctx.read_json("data/data_version.json")
                   ["slices"] if x in test.columns]
    table, gaps = bias_stage.slice_report(
        test[["nct_id", "label", *slice_names]], {cand["name"]: s},
        slice_names, ev["threshold"], b)
    worst = bias_stage.worst_gaps(gaps, [cand["name"]], b["metrics"])
    m["max_bias_gap"] = worst[cand["name"]]["max_gap"]
    m["max_bias_gap_slice"] = worst[cand["name"]]["slice"]

    out = ctx.path("reports", "test", ".keep").parent
    ctx.write_json("reports/test/test_metrics.json", m)
    table.to_csv(out / "test_model_slices.csv", index=False)
    plots.confusion(test["label"], s, ev["threshold"],
                    out / "test_confusion.png", f"{cand['name']} (test)")
    plots.pr_curves({cand["name"]: (test["label"].to_numpy(), s)},
                    out / "test_pr_curve.png", "Precision-recall, test set")
    tracking.log_metrics(ctx, m, prefix="test_")
    tracking.log_artifacts(ctx, out, "test")

    problems = []
    if (m["pr_auc"] or 0) < gate["min_pr_auc"]:
        problems.append(f"test PR-AUC {m['pr_auc']:.3f}")
    if (m["roc_auc"] or 0) < gate["min_roc_auc"]:
        problems.append(f"test ROC-AUC {m['roc_auc']:.3f}")
    if problems:
        notify.alert(ctx, "selected model failed on the test set",
                     ", ".join(problems))
        raise SystemExit("selected model failed the gate on the test set")

    art = ctx.path("artifact", ".keep").parent
    shutil.copy(ctx.path(cand["model_file"]), art / "model.joblib")
    ctx.write_json("artifact/model_card.json", {
        "candidate": cand, "feature_columns": columns,
        "threshold": ev["threshold"], "test_metrics": m,
        "val_metrics": ctx.read_json("reports/val_metrics.json")
        [cand["name"]],
        "data_version": ctx.read_json("data/data_version.json"),
        "run_id": ctx.state.get("run_id")})
    logger.info("test PR-AUC %.4f ROC-AUC %.4f", m["pr_auc"], m["roc_auc"])
    return m
