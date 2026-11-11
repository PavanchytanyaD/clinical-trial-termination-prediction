"""Stage select: choose the final model after both validation and bias.

A candidate is eligible if its validation PR-AUC and ROC-AUC reach the gate
(params.yaml, gate) and its largest subgroup gap is not above
block_bias_gap. Among eligible candidates the one with the highest
    score = PR-AUC - bias_penalty * largest gap
wins, so a slightly less accurate but much fairer model can win. If no
candidate is eligible the run stops here (nothing is registered) and an
alert is sent.
"""

import logging

from model_pipeline import notify, tracking

logger = logging.getLogger(__name__)


def choose(rows, gate):
    """rows: {name: {pr_auc, roc_auc, max_gap}} -> (winner or None, table)."""
    table = []
    for name, r in rows.items():
        reasons = []
        if (r["pr_auc"] or 0) < gate["min_pr_auc"]:
            reasons.append(f"PR-AUC {r['pr_auc']:.3f} < {gate['min_pr_auc']}")
        if (r["roc_auc"] or 0) < gate["min_roc_auc"]:
            reasons.append(
                f"ROC-AUC {r['roc_auc']:.3f} < {gate['min_roc_auc']}")
        if r["max_gap"] > gate["block_bias_gap"]:
            reasons.append(f"bias gap {r['max_gap']:.3f} > "
                           f"{gate['block_bias_gap']}")
        score = (r["pr_auc"] or 0) - gate["bias_penalty"] * r["max_gap"]
        table.append({"name": name, **r, "score": score,
                      "eligible": not reasons, "reasons": reasons})
    table.sort(key=lambda t: t["score"], reverse=True)
    winner = next((t for t in table if t["eligible"]), None)
    return winner, table


def markdown(winner, table, gate):
    lines = ["# Model selection", "",
             f"Score = validation PR-AUC - {gate['bias_penalty']} x largest "
             f"subgroup gap. Gate: PR-AUC >= {gate['min_pr_auc']}, ROC-AUC "
             f">= {gate['min_roc_auc']}, gap <= {gate['block_bias_gap']}.",
             "", "| Candidate | PR-AUC | ROC-AUC | Largest gap | Score | "
             "Eligible |", "|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['name']} | {t['pr_auc']:.3f} | "
                     f"{t['roc_auc']:.3f} | {t['max_gap']:.3f} | "
                     f"{t['score']:.3f} | "
                     f"{'yes' if t['eligible'] else '; '.join(t['reasons'])}"
                     " |")
    lines += ["", f"**Selected: {winner['name']}**" if winner
              else "**No candidate passed the gate.**"]
    return "\n".join(lines) + "\n"


def run(ctx):
    gate = ctx.params["gate"]
    val = ctx.read_json("reports/val_metrics.json")
    bias = ctx.read_json("reports/bias/bias_summary.json")
    rows = {name: {"pr_auc": val[name]["pr_auc"],
                   "roc_auc": val[name]["roc_auc"],
                   "max_gap": bias[name]["max_gap"],
                   "gap_slice": bias[name]["slice"]}
            for name in bias}
    winner, table = choose(rows, gate)
    ctx.write_json("reports/selection.json",
                   {"selected": winner, "candidates": table})
    ctx.path("reports", "selection.md").write_text(
        markdown(winner, table, gate))
    tracking.log_artifacts(ctx, ctx.path("reports"), "reports")

    if winner is None:
        notify.alert(ctx, "no model passed validation and bias checks",
                     markdown(winner, table, gate))
        raise SystemExit("no candidate passed the gate; nothing registered")
    tracking.set_tags(ctx, {"selected_model": winner["name"]})
    if winner["max_gap"] > gate["alert_bias_gap"]:
        notify.alert(ctx, f"bias flagged for {winner['name']}",
                     f"Largest gap {winner['max_gap']:.3f} "
                     f"({winner['gap_slice']}), above the alert level "
                     f"{gate['alert_bias_gap']} but under the block level. "
                     "See reports/bias/bias_report.md.")
    logger.info("selected %s (score %.4f)", winner["name"], winner["score"])
    return winner
