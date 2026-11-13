"""Stage test: the selected model on the test set, the most recently
registered trials, which no earlier stage has looked at. The model must
pass the same gate here, then it is packaged for the registry:
artifact/model.joblib (the trained model), artifact/calibrator.joblib (raw
score -> probability, from the calibrate stage), artifact/text_pca.joblib
(BioBERT vectors -> text features, for a model with text features) and
artifact/model_card.json (features, threshold, group thresholds,
calibration, metrics and the data version). Writes data/test_scores.pkl for
the lead_time stage."""

import logging
import shutil

import joblib
import numpy as np

from model_pipeline import bias as bias_stage
from model_pipeline import calibrate, data, metrics, notify, plots, tracking
from model_pipeline.models import score_candidate
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


def run(ctx):
    ev, gate, b = (ctx.params["evaluation"], ctx.params["gate"],
                   ctx.params["bias"])
    test = data.load_split(ctx, "test")
    cand = calibrate.selected_candidate(ctx)
    columns = data.load_columns(ctx, cand)
    model = load_model(ctx, cand)
    s = score_candidate(cand, model, test, columns, ev["threshold"])
    m = metrics.compute(test["label"], s, ev["threshold"], ev["top_k_share"])
    calibrator = joblib.load(ctx.path("models", "calibrator.joblib"))
    p = calibrate.apply(calibrator, s)
    y = test["label"].to_numpy().astype(int)
    m.update(brier_calibrated=float(np.mean((p - y) ** 2)),
             ece_raw=metrics.expected_calibration_error(y, s),
             ece_calibrated=metrics.expected_calibration_error(y, p))

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
    plots.reliability({"raw score": (y, s), "calibrated": (y, p)},
                      out / "test_reliability.png",
                      f"Reliability, test set ({cand['name']})")
    test.assign(score=s, probability=p,
                flagged=s >= ev["threshold"]).drop(
        columns=columns).to_pickle(ctx.path("data", "test_scores.pkl"))
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
    shutil.copy(ctx.path("models", "calibrator.joblib"),
                art / "calibrator.joblib")
    if cand.get("text_features"):
        shutil.copy(ctx.path("data", "text_pca.joblib"),
                    art / "text_pca.joblib")
    ctx.write_json("artifact/model_card.json", {
        "candidate": cand, "feature_columns": columns,
        "threshold": ev["threshold"], "test_metrics": m,
        "calibration": {
            **ctx.read_json("reports/calibration/calibration.json"),
            "usage": "flag a trial when the model score reaches threshold; "
                     "report calibrator.joblib applied to that score as "
                     "the probability of stopping"},
        "text_features": ({
            "embedding_model": "dmis-lab/biobert-base-cased-v1.2",
            "reducers": "text_pca.joblib (one PCA per field, fitted on the "
                        "training split)",
            "fields": ctx.params["data"]["embeddings"]["fields"]}
            if cand.get("text_features") else None),
        "val_metrics": ctx.read_json("reports/val_metrics.json")
        [cand["name"]],
        "data_version": ctx.read_json("data/data_version.json"),
        "run_id": ctx.state.get("run_id")})
    logger.info("test PR-AUC %.4f ROC-AUC %.4f", m["pr_auc"], m["roc_auc"])
    return m
