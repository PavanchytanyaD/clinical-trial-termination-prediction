"""Stage bias: performance of every candidate per subgroup (Fairlearn),
with the data pipeline's slice code (src/bias/model_slices.py).

Slices: phase, sponsor class, region, therapeutic area, gender, age group,
start era. For each candidate, the gap of a metric is the best group minus
the worst group (groups with at least min_group_size validation trials and
min_stopped stopped ones, UNKNOWN left out). Mitigation candidates:
  - reweighted (from the train stage): trained with the trial weights;
  - group_thresholds (added here): one decision threshold per region,
    chosen on the training set so every region reaches the same recall.
Outputs in reports/bias/: model_slices.csv, model_gaps.csv,
bias_summary.json, bias_report.md and gap charts.
"""

import logging

import pandas as pd

from model_pipeline import data, metrics, plots, tracking
from model_pipeline.models import predict, score_candidate
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


def add_group_threshold_candidates(ctx, candidates, train, columns):
    """One group_thresholds candidate per unmitigated model."""
    from scripts.bias_model import group_thresholds

    b, threshold = ctx.params["bias"], ctx.params["evaluation"]["threshold"]
    if b["threshold_slice"] not in train.columns:
        return candidates
    extra = []
    for cand in candidates:
        if cand["mitigation"] != "none":
            continue
        s = predict(load_model(ctx, cand),
                    train[columns].to_numpy(dtype=float))
        cuts = group_thresholds(s, train["label"].to_numpy(),
                                train[b["threshold_slice"]].to_numpy(),
                                threshold, b["min_stopped"])
        if cuts:
            extra.append({**cand, "name": f"{cand['family']}"
                          "+group_thresholds",
                          "mitigation": "group_thresholds",
                          "threshold_slice": b["threshold_slice"],
                          "cuts": cuts})
    return candidates + extra


def slice_report(rows, scores, slice_names, threshold, b):
    """Per-group table and gaps for a dict of {candidate: scores}."""
    from scripts.bias_model import gaps_by_slice
    from src.bias.model_slices import model_slice_report

    tables = []
    for name, s in scores.items():
        table, _ = model_slice_report(rows.assign(y_score=s), slice_names,
                                      threshold, b["min_group_size"])
        tables.append(table.assign(model=name))
    table = pd.concat(tables, ignore_index=True)
    return table, gaps_by_slice(table, b["min_group_size"],
                                b["min_stopped"])


def worst_gaps(gaps, names, bias_metrics):
    """Largest gap of each candidate over slices and checked metrics."""
    out = {}
    for name in names:
        g = gaps[gaps["model"] == name]
        best = {"max_gap": 0.0, "slice": None, "metric": None}
        for row in g.itertuples():
            for m in bias_metrics:
                v = getattr(row, m)
                if pd.notna(v) and v > best["max_gap"]:
                    best = {"max_gap": float(v), "slice": row.slice,
                            "metric": m}
        best["n_slices_checked"] = int(len(g))
        out[name] = best
    return out


def markdown(summary, gaps, b, gate):
    lines = ["# Model bias across subgroups (validation set)", "",
             f"Gap = best group minus worst group, over groups with at "
             f"least {b['min_group_size']} trials and {b['min_stopped']} "
             f"stopped ones (UNKNOWN left out). Alert above "
             f"{gate['alert_bias_gap']}, block above "
             f"{gate['block_bias_gap']}.", "",
             "| Candidate | Largest gap | Slice | Metric | PR-AUC |",
             "|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['max_gap']:.3f} | {s['slice']} | "
                     f"{s['metric']} | {s['pr_auc']:.3f} |")
    lines += ["", "## Gap per slice", "",
              "| Slice | Candidate | Groups | " + " | ".join(b["metrics"])
              + " |", "|---|---|---|" + "---|" * len(b["metrics"])]
    for row in gaps.itertuples():
        vals = [getattr(row, m) for m in b["metrics"]]
        lines.append(f"| {row.slice} | {row.model} | {row.n_groups_used} | "
                     + " | ".join("n/a" if pd.isna(v) else f"{v:.3f}"
                                  for v in vals) + " |")
    lines += ["", "## Mitigation", "",
              "- **reweighted**: trained with the bias stage's trial weights "
              "(under-represented groups weighted up).",
              "- **group_thresholds**: one decision threshold per "
              f"{b['threshold_slice']} group, chosen on the training set "
              "for equal recall; it trades some precision for a smaller "
              "recall gap.",
              "- The select stage weighs PR-AUC against the largest gap, so "
              "the final choice is made after this check."]
    return "\n".join(lines) + "\n"


def run(ctx):
    b, gate = ctx.params["bias"], ctx.params["gate"]
    ev = ctx.params["evaluation"]
    columns = data.load_columns(ctx)
    train, val = data.load_split(ctx, "train"), data.load_split(ctx, "val")
    slice_names = [s for s in ctx.read_json("data/data_version.json")["slices"]
                   if s in val.columns]

    candidates = add_group_threshold_candidates(
        ctx, ctx.read_json("candidates.json"), train, columns)
    ctx.write_json("candidates.json", candidates)

    scores, val_metrics = {}, ctx.read_json("reports/val_metrics.json")
    for cand in candidates:
        s = score_candidate(cand, load_model(ctx, cand), val, columns,
                            ev["threshold"])
        scores[cand["name"]] = s
        if cand["name"] not in val_metrics:
            val_metrics[cand["name"]] = metrics.compute(
                val["label"], s, ev["threshold"], ev["top_k_share"])
    ctx.write_json("reports/val_metrics.json", val_metrics)

    rows = val[["nct_id", "label", *slice_names]]
    table, gaps = slice_report(rows, scores, slice_names, ev["threshold"], b)
    summary = worst_gaps(gaps, list(scores), b["metrics"])
    for name in summary:
        summary[name]["pr_auc"] = val_metrics[name]["pr_auc"]
        summary[name]["flagged"] = \
            summary[name]["max_gap"] > gate["alert_bias_gap"]

    out = ctx.path("reports", "bias", ".keep").parent
    table.to_csv(out / "model_slices.csv", index=False)
    gaps.to_csv(out / "model_gaps.csv", index=False)
    ctx.write_json("reports/bias/bias_summary.json", summary)
    (out / "bias_report.md").write_text(markdown(summary, gaps, b, gate))
    for m in b["metrics"]:
        plots.bias_gaps(gaps, m, out / f"gap_{m}.png")
    tracking.log_artifacts(ctx, out, "bias")
    for name, s in summary.items():
        tracking.log_metrics(ctx, {"max_bias_gap": s["max_gap"]},
                             prefix=f"val_{name}_")
        if s["flagged"]:
            logger.warning("%s: %s %s gap %.3f above %.2f", name,
                           s["slice"], s["metric"], s["max_gap"],
                           gate["alert_bias_gap"])
    return summary

