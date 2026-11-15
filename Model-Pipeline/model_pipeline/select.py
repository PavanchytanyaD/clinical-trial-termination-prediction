"""Stage select: choose the final model after both validation and bias.

A candidate is eligible if its validation PR-AUC and ROC-AUC reach the gate
(params.yaml, gate) and the bias stage did not block it (a gating slice
clearly above block_bias_gap, or no gating slice with enough data when
insufficient_bias_data is "block"). Eligible candidates are ranked by
    score = PR-AUC - bias_penalty * largest gating-slice gap
(the gap is the point estimate of overall minus worst group, the largest
over the gating slices and bias metrics), so a slightly less accurate but
fairer model can win. The test stage tries them in this order. If no
candidate is eligible the run stops here (nothing is registered) and an
alert is sent.
"""

import logging

from model_pipeline import notify, tracking

logger = logging.getLogger(__name__)


def choose(rows, gate):
    """rows: {name: {pr_auc, roc_auc, max_gap, bias_blocked, ...}} ->
    (winner or None, table ranked: eligible first, by score)."""
    table = []
    for name, r in rows.items():
        reasons = []
        if (r["pr_auc"] or 0) < gate["min_pr_auc"]:
            reasons.append(f"PR-AUC {r['pr_auc'] or 0:.3f} < "
                           f"{gate['min_pr_auc']}")
        if (r["roc_auc"] or 0) < gate["min_roc_auc"]:
            reasons.append(f"ROC-AUC {r['roc_auc'] or 0:.3f} < "
                           f"{gate['min_roc_auc']}")
        if r.get("bias_blocked"):
            reasons.append(f"bias: {r.get('bias_status', 'block')}")
        score = (r["pr_auc"] or 0) - gate["bias_penalty"] * r["max_gap"]
        table.append({"name": name, **r, "score": score,
                      "eligible": not reasons, "reasons": reasons})
    table.sort(key=lambda t: (t["eligible"], t["score"]), reverse=True)
    for rank, t in enumerate(table, 1):
        t["rank"] = rank if t["eligible"] else None
    winner = table[0] if table and table[0]["eligible"] else None
    return winner, table


def markdown(winner, table, gate, gating, tries=None, deployed=None):
    lines = ["# Model selection", "",
             "How the final model is chosen:", "",
             f"1. **Validation gate**: PR-AUC >= {gate['min_pr_auc']} and "
             f"ROC-AUC >= {gate['min_roc_auc']}.",
             "2. **Bias gate** (reports/bias/bias_report.md): not blocked, "
             f"i.e. no gating slice ({', '.join(gating)}) whose worst "
             "group is clearly (lower bound of the "
             f"bootstrap interval) more than {gate['block_bias_gap']} below "
             "overall, and at least one gating slice with enough data.",
             f"3. **Ranking**: score = validation PR-AUC - "
             f"{gate['bias_penalty']} x largest gating-slice gap (point "
             "estimate). Accuracy and fairness are traded explicitly: 0.1 "
             f"of gap costs {gate['bias_penalty'] * 0.1:.2f} of PR-AUC.",
             "4. **Test set**: the top candidate is checked on the untouched "
             "test set with the same validation and bias gates. If it fails, "
             f"the next eligible one is tried (at most "
             f"{gate.get('max_test_tries', 3)}); the run stops only if none "
             "passes.", "",
             "| Rank | Candidate | PR-AUC | ROC-AUC | Largest gating gap | "
             "Bias status | Score | Eligible |",
             "|---|---|---|---|---|---|---|---|"]
    for t in table:
        gap = (f"{t['max_gap']:.3f} ({t['gap_slice']} {t.get('gap_metric')})"
               if t.get("gap_slice") else "n/a")
        lines.append(
            f"| {t['rank'] or '-'} | {t['name']} | {t['pr_auc'] or 0:.3f} | "
            f"{t['roc_auc'] or 0:.3f} | {gap} | "
            f"{t.get('bias_status', '-')} | {t['score']:.3f} | "
            f"{'yes' if t['eligible'] else '; '.join(t['reasons'])} |")
    lines += ["", f"**Best on validation: {winner['name']}**" if winner
              else "**No candidate passed the gate.**"]
    if tries:
        lines += ["", "## Test set", "",
                  "| Try | Candidate | PR-AUC | ROC-AUC | Largest gating gap "
                  "| Bias status | Result |", "|---|---|---|---|---|---|---|"]
        for i, t in enumerate(tries, 1):
            lines.append(
                f"| {i} | {t['name']} | {t['pr_auc'] or 0:.3f} | "
                f"{t['roc_auc'] or 0:.3f} | {t['max_bias_gap']:.3f} | "
                f"{t['bias_status']} | "
                f"{'passed' if not t['problems'] else '; '.join(t['problems'])}"
                " |")
        lines += ["", f"**Final model: {deployed}**" if deployed else
                  "**No candidate passed on the test set; nothing is "
                  "registered.**"]
    return "\n".join(lines) + "\n"


def run(ctx):
    gate, gating = ctx.params["gate"], ctx.params["bias"]["gating_slices"]
    val = ctx.read_json("reports/val_metrics.json")
    bias = ctx.read_json("reports/bias/bias_summary.json")
    rows = {name: {"pr_auc": val[name]["pr_auc"],
                   "roc_auc": val[name]["roc_auc"],
                   "max_gap": bias[name]["max_gap"],
                   "gap_slice": bias[name]["slice"],
                   "gap_metric": bias[name]["metric"],
                   "bias_status": bias[name]["status"],
                   "bias_blocked": bias[name]["blocked"]}
            for name in bias}
    winner, table = choose(rows, gate)
    ctx.write_json("reports/selection.json",
                   {"selected": winner, "candidates": table})
    ctx.path("reports", "selection.md").write_text(
        markdown(winner, table, gate, gating))
    tracking.log_artifacts(ctx, ctx.path("reports"), "reports")

    if winner is None:
        notify.alert(ctx, "no model passed validation and bias checks",
                     markdown(winner, table, gate, gating))
        raise SystemExit("no candidate passed the gate; nothing registered")
    tracking.set_tags(ctx, {"selected_model": winner["name"]})
    if winner["bias_status"] != "ok":
        detail = "; ".join(bias[winner["name"]]["alerts"]) or \
            winner["bias_status"]
        notify.alert(ctx, f"bias flagged for {winner['name']}",
                     f"Selected with bias status {winner['bias_status']}: "
                     f"{detail}. Largest gating gap {winner['max_gap']:.3f} "
                     f"({winner['gap_slice']}); alert level "
                     f"{gate['alert_bias_gap']}, block level "
                     f"{gate['block_bias_gap']}. See "
                     "reports/bias/bias_report.md.")
    logger.info("selected %s (score %.4f)", winner["name"], winner["score"])
    return winner
