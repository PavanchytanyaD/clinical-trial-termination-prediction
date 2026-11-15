"""Stage test: the selected model on the test set, the most recently
registered trials, which no earlier stage has looked at.

The model must pass the same gate here: PR-AUC and ROC-AUC, and the bias
gate (no gating slice clearly above block_bias_gap, gaps computed as in the
bias stage, now on the test trials). If it fails, the next candidate in the
select stage's ranking is tried, up to gate.max_test_tries candidates; the
run stops only when none of them passes. A candidate other than the one the
calibrate stage calibrated gets its own calibrator (same method choice, on
the validation set).

The model that passes is packaged for the registry: artifact/model.joblib
(the trained model), artifact/calibrator.joblib (raw score -> probability),
artifact/text_pca.joblib (BioBERT vectors -> text features, for a model
with text features) and artifact/model_card.json (features, threshold,
group thresholds, calibration, metrics, test bias, the candidates tried and
the data version). Writes data/test_scores.pkl for the lead_time stage."""

import logging
import shutil

import joblib
import numpy as np

from model_pipeline import bias as bias_stage
from model_pipeline import calibrate, data, metrics, notify, plots, tracking
from model_pipeline.models import score_candidate
from model_pipeline.select import markdown as selection_markdown
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


def gate_problems(m, decision, gate):
    """Reasons a model fails the gate on the test set (empty: passes)."""
    problems = []
    if (m["pr_auc"] or 0) < gate["min_pr_auc"]:
        problems.append(f"test PR-AUC {m['pr_auc'] or 0:.3f} < "
                        f"{gate['min_pr_auc']}")
    if (m["roc_auc"] or 0) < gate["min_roc_auc"]:
        problems.append(f"test ROC-AUC {m['roc_auc'] or 0:.3f} < "
                        f"{gate['min_roc_auc']}")
    if decision["blocked"]:
        problems.append("test bias: " + ("; ".join(decision["blocks"])
                                         or decision["status"]))
    return problems


def evaluate(ctx, cand, test):
    """Scores, metrics, bias tables and gate problems of one candidate."""
    ev, gate = ctx.params["evaluation"], ctx.params["gate"]
    columns = data.load_columns(ctx, cand)
    s = score_candidate(cand, load_model(ctx, cand), test, columns,
                        ev["threshold"])
    m = metrics.compute(test["label"], s, ev["threshold"], ev["top_k_share"])
    table, groups, slices, decisions = bias_stage.assess_split(
        ctx, test, {cand["name"]: s})
    d = decisions[cand["name"]]
    m.update(max_bias_gap=d["max_gap"], max_bias_gap_slice=d["slice"],
             max_bias_gap_metric=d["metric"], bias_status=d["status"])
    return {"scores": s, "metrics": m, "table": table, "groups": groups,
            "slices": slices, "decision": d, "columns": columns,
            "problems": gate_problems(m, d, gate)}


def calibrator_for(ctx, cand):
    """The calibrate stage's calibrator when it was fitted for cand, else a
    new one for cand (the test stage fell back to another candidate)."""
    done = ctx.read_json("reports/calibration/calibration.json")
    if done.get("candidate") != cand["name"]:
        logger.info("calibrating %s (the calibrate stage calibrated %s)",
                    cand["name"], done.get("candidate"))
        calibrate.calibrate(ctx, cand)
    return joblib.load(ctx.path("models", "calibrator.joblib"))


def run(ctx):
    ev, gate = ctx.params["evaluation"], ctx.params["gate"]
    test = data.load_split(ctx, "test")
    selection = ctx.read_json("reports/selection.json")
    by_name = {c["name"]: c for c in ctx.read_json("candidates.json")}
    ranked = [t for t in selection["candidates"] if t["eligible"]]
    ranked = ranked[:int(gate.get("max_test_tries", 3))]

    tries, chosen = [], None
    for t in ranked:
        cand = by_name[t["name"]]
        result = evaluate(ctx, cand, test)
        m = result["metrics"]
        tries.append({"name": cand["name"], "pr_auc": m["pr_auc"],
                      "roc_auc": m["roc_auc"],
                      "max_bias_gap": m["max_bias_gap"],
                      "bias_status": m["bias_status"],
                      "problems": result["problems"]})
        logger.info("test %s: PR-AUC %.4f ROC-AUC %.4f bias %s",
                    cand["name"], m["pr_auc"] or 0, m["roc_auc"] or 0,
                    m["bias_status"])
        if not result["problems"]:
            chosen = (cand, result)
            break
        logger.warning("%s failed on the test set: %s", cand["name"],
                       "; ".join(result["problems"]))
        notify.alert(ctx, f"{cand['name']} failed on the test set",
                     "; ".join(result["problems"]) + "\n\nTrying the next "
                     "candidate in the ranking, if any.")

    deployed = chosen[0]["name"] if chosen else None
    selection["test_tries"] = tries
    selection.setdefault("selected_on_validation", selection["selected"])
    if chosen:
        selection["selected"] = next(t for t in selection["candidates"]
                                     if t["name"] == deployed)
    ctx.write_json("reports/selection.json", selection)
    ctx.path("reports", "selection.md").write_text(selection_markdown(
        selection["selected_on_validation"], selection["candidates"], gate,
        ctx.params["bias"]["gating_slices"], tries, deployed))

    out = ctx.path("reports", "test", ".keep").parent
    ctx.write_json("reports/test/test_tries.json", tries)
    if chosen is None:
        notify.alert(ctx, "no candidate passed on the test set",
                     ctx.path("reports", "selection.md").read_text())
        raise SystemExit("no candidate passed the gate on the test set")

    cand, result = chosen
    m, s, columns = result["metrics"], result["scores"], result["columns"]
    if len(tries) > 1:
        notify.alert(ctx, f"fell back to {deployed} on the test set",
                     f"{tries[0]['name']} failed on the test set; "
                     f"{deployed} (rank {len(tries)}) passed and is "
                     "packaged.")
    if result["decision"]["status"] != "ok":
        notify.alert(ctx, f"test-set bias flagged for {deployed}",
                     f"Bias status {result['decision']['status']}: "
                     + "; ".join(result["decision"]["alerts"]))

    calibrator = calibrator_for(ctx, cand)
    p = calibrate.apply(calibrator, s)
    y = test["label"].to_numpy().astype(int)
    m.update(brier_calibrated=float(np.mean((p - y) ** 2)),
             ece_raw=metrics.expected_calibration_error(y, s),
             ece_calibrated=metrics.expected_calibration_error(y, p))

    ctx.write_json("reports/test/test_metrics.json", m)
    result["table"].to_csv(out / "test_model_slices.csv", index=False)
    result["groups"].to_csv(out / "test_group_gaps.csv", index=False)
    result["slices"].to_csv(out / "test_gaps.csv", index=False)
    for metric in ctx.params["bias"]["metrics"]:
        plots.bias_gap_bars(result["slices"], metric, [deployed],
                            out / f"test_gap_{metric}.png",
                            f"{metric} gap per slice, {deployed} (test set)",
                            gate["alert_bias_gap"], gate["block_bias_gap"])
    plots.confusion(test["label"], s, ev["threshold"],
                    out / "test_confusion.png", f"{deployed} (test)")
    plots.pr_curves({deployed: (test["label"].to_numpy(), s)},
                    out / "test_pr_curve.png", "Precision-recall, test set")
    plots.reliability({"raw score": (y, s), "calibrated": (y, p)},
                      out / "test_reliability.png",
                      f"Reliability, test set ({deployed})")
    test.assign(score=s, probability=p,
                flagged=s >= ev["threshold"]).drop(
        columns=columns).to_pickle(ctx.path("data", "test_scores.pkl"))
    tracking.log_metrics(ctx, m, prefix="test_")
    tracking.set_tags(ctx, {"deployed_model": deployed,
                            "test_tries": str(len(tries))})
    tracking.log_artifacts(ctx, out, "test")

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
        "test_bias": result["decision"],
        "val_metrics": ctx.read_json("reports/val_metrics.json")
        [cand["name"]],
        "test_tries": tries,
        "data_version": ctx.read_json("data/data_version.json"),
        "run_id": ctx.state.get("run_id")})
    logger.info("test PR-AUC %.4f ROC-AUC %.4f (%s)", m["pr_auc"],
                m["roc_auc"], deployed)
    return m
