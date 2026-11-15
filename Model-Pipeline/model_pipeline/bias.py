"""Stage bias: performance of every candidate per subgroup, on the
validation set, and the bias gate the select stage uses.

Per-group metrics come from Fairlearn with the data pipeline's slice code
(src/bias/model_slices.py). The gate (model_pipeline/gaps.py) compares the
worst qualified group of each slice with the trials overall, with a
bootstrap interval, so that only clear gaps count:
  - gating slices (params.yaml bias.gating_slices; region, phase, sponsor
    class) can block a model;
  - report slices (bias.report_slices) are reported and can raise an alert,
    but do not block.

Mitigation candidates:
  - reweighted (from the train stage): trained with the trial weights;
  - group_thresholds (added here): one decision threshold per group of a
    slice (bias.threshold_slices, e.g. region, or region then phase), so
    every group reaches the model's overall recall. The cuts are chosen on
    the validation set, never on the training set (where a boosted model's
    scores are too good). To score them honestly on the same validation
    set they are cross-fitted: the set is split into threshold_folds parts,
    the cuts are fitted on the other parts and applied to each part. The
    cuts kept for the test set and deployment are fitted on all of it.

Outputs in reports/bias/: model_slices.csv (Fairlearn, every group),
model_group_gaps.csv (every group's gap and interval), model_gaps.csv (one
row per slice and metric), bias_summary.json, bias_report.md, gap charts.
"""

import logging

import numpy as np
import pandas as pd

from model_pipeline import data, gaps, metrics, notify, plots, tracking
from model_pipeline.models import score_candidate, shift_for_groups
from model_pipeline.train import load_model

logger = logging.getLogger(__name__)


# ---------- per-group thresholds, fitted on validation scores ----------

def fit_threshold_steps(scores, labels, slices, names, threshold,
                        min_stopped):
    """Cuts for each slice in names, fitted one after the other (the second
    slice's cuts are fitted on the scores already moved by the first)."""
    from scripts.bias_model import group_thresholds

    s, steps = np.asarray(scores, dtype=float), []
    for name in names:
        cuts = group_thresholds(s, labels, slices[name], threshold,
                                min_stopped)
        steps.append({"slice": name, "cuts": cuts})
        s = shift_for_groups(s, slices[name], cuts, threshold)
    return steps


def apply_threshold_steps(scores, slices, steps, threshold):
    s = np.asarray(scores, dtype=float)
    for step in steps:
        s = shift_for_groups(s, slices[step["slice"]], step["cuts"],
                             threshold)
    return s


def fold_ids(labels, groups, n_folds, seed):
    """Fold of each trial, balanced within every (label, group) pair."""
    rng = np.random.default_rng(seed)
    key = pd.Series(labels).astype(str).to_numpy() + "|" + \
        np.asarray(groups, dtype=str)
    folds = np.zeros(len(key), dtype=int)
    for k in pd.unique(key):
        idx = np.flatnonzero(key == k)
        rng.shuffle(idx)
        folds[idx] = (np.arange(len(idx)) + rng.integers(n_folds)) % n_folds
    return folds


def crossfit_thresholds(scores, labels, slices, names, threshold,
                        min_stopped, n_folds, seed):
    """(out-of-fold scores, cuts fitted on all trials).

    Each trial's moved score comes from cuts fitted without it, so metrics
    on these scores are not flattered by the fit."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    folds = fold_ids(labels, slices[names[0]], n_folds, seed)
    oof = scores.copy()
    for k in range(n_folds):
        fit, held = folds != k, folds == k
        part = {n: v[fit] for n, v in slices.items()}
        steps = fit_threshold_steps(scores[fit], labels[fit], part, names,
                                    threshold, min_stopped)
        oof[held] = apply_threshold_steps(
            scores[held], {n: v[held] for n, v in slices.items()}, steps,
            threshold)
    return oof, fit_threshold_steps(scores, labels, slices, names, threshold,
                                    min_stopped)


def model_entry(cand):
    """The models entry a candidate comes from ("xgboost_text" for
    "xgboost_text+reweighted"); several entries can share a family."""
    return cand["name"].split("+")[0]


def threshold_sets(b):
    """bias.threshold_slices as lists: "region" -> [region]."""
    return [[x] if isinstance(x, str) else list(x)
            for x in b.get("threshold_slices") or []]


def add_threshold_candidates(b, candidates, scores, val, threshold, seed):
    """One group_thresholds candidate per unmitigated model and threshold
    set. Returns (new candidates, their out-of-fold validation scores)."""
    labels = val["label"].to_numpy()
    extra, oof = [], {}
    for names in threshold_sets(b):
        bad = [n for n in names if n not in val.columns
               or val[n].str.contains("|", regex=False).any()]
        if bad:      # one group per trial is needed to move its score
            logger.warning("thresholds on %s skipped: %s missing or "
                           "multi-valued", names, bad)
            continue
        slices = {n: val[n].to_numpy(dtype=str) for n in names}
        for cand in candidates:
            if cand["mitigation"] != "none":
                continue
            s_oof, steps = crossfit_thresholds(
                scores[cand["name"]], labels, slices, names, threshold,
                b["min_stopped"], int(b.get("threshold_folds", 2)), seed)
            if not any(step["cuts"] for step in steps):
                continue
            name = f"{cand['name']}+thresholds_{'_'.join(names)}"
            extra.append({k: v for k, v in cand.items()
                          if k not in ("cuts", "threshold_slice")}
                         | {"name": name, "base": cand["name"],
                            "mitigation": "group_thresholds",
                            "threshold_slices": names,
                            "threshold_steps": steps,
                            "thresholds_fitted_on": "validation"})
            oof[name] = s_oof
    return extra, oof


# ---------- gaps on one split (also used by the test stage) ----------

def slice_roles(ctx, df):
    """(all slices to check, gating slices) present in this data."""
    b = ctx.params["bias"]
    have = [s for s in ctx.read_json("data/data_version.json")["slices"]
            if s in df.columns]
    gating = [s for s in b["gating_slices"] if s in have]
    report = [s for s in b.get("report_slices") or []
              if s in have and s not in gating]
    return gating + report, gating


def fairlearn_table(rows, scores, slice_names, threshold, min_group_size):
    """Per-group metrics of every candidate (Fairlearn MetricFrame)."""
    from src.bias.model_slices import model_slice_report

    tables = []
    for name, s in scores.items():
        table, _ = model_slice_report(rows.assign(y_score=s), slice_names,
                                      threshold, min_group_size)
        tables.append(table.assign(model=name))
    return pd.concat(tables, ignore_index=True)


def assess_split(ctx, df, scores):
    """(Fairlearn table, group gaps, slice gaps, {name: decision})."""
    b, gate = ctx.params["bias"], ctx.params["gate"]
    threshold = ctx.params["evaluation"]["threshold"]
    names, gating = slice_roles(ctx, df)
    rows = df[["nct_id", "label", *names]]
    table = fairlearn_table(rows, scores, names, threshold,
                            b["min_group_size"])
    groups, slices, decisions = gaps.assess(
        rows, scores, names, gating, threshold, b, gate,
        int(ctx.params["seed"]))
    return table, groups, slices, decisions


# ---------- report ----------

def f3(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}"


def gap_cell(r):
    if r.status == "insufficient data":
        return f"n/a ({r.n_qualified} qualified)"
    return (f"{f3(r.gap)} [{f3(r.ci_low)}, {f3(r.ci_high)}] "
            f"({r.worst_group})")


def rules_text(b, gate, gating, report, split="validation"):
    level = int(round(100 * float(b.get("ci_level", 0.9))))
    return [
        f"- **Gap** of a metric for a group = overall value minus the "
        f"group's value on the {split} set (positive: the group is served "
        f"worse). Metrics: {', '.join(b['metrics'])} (recall at the "
        "decision threshold).",
        f"- **Qualified groups**: at least {b['min_group_size']} trials and "
        f"{b['min_stopped']} stopped ones, not UNKNOWN. A slice is checked "
        f"only if it has at least {b.get('min_groups', 2)} qualified groups;"
        " its gap is that of its worst group.",
        f"- **Interval**: {level}% percentile bootstrap, "
        f"{b.get('bootstrap', 200)} resamples of the trials (seeded). A gap "
        "counts as above a level only if its lower bound is above it.",
        f"- **Gating slices** (can block): {', '.join(gating) or 'none'}. "
        f"**Report slices** (alert only): {', '.join(report) or 'none'}.",
        f"- **Block**: a gating slice above {gate['block_bias_gap']}. "
        f"**Alert**: any slice above {gate['alert_bias_gap']}. "
        "**Insufficient data**: no gating slice could be checked; the "
        f"candidate is then treated as "
        f"`{gate.get('insufficient_bias_data', 'block')}`.",
    ]


def markdown(ctx, summary, slices, groups, val_metrics, candidates, val):
    b, gate = ctx.params["bias"], ctx.params["gate"]
    names, gating = slice_roles(ctx, val)
    report = [s for s in names if s not in gating]
    lines = ["# Model bias across subgroups (validation set)", "",
             f"{len(val)} validation trials, {int(val['label'].sum())} "
             "stopped. Every candidate is checked on the same trials; the "
             "threshold candidates are scored out of fold (see "
             "Mitigation).", "", "## Gate rule", "",
             *rules_text(b, gate, gating, report),
             "- **Why worst group against overall**: the best-minus-worst "
             "spread grows with the number of groups by chance alone and "
             "often hinges on one small lucky group; against overall the "
             "number says how much worse the worst group is served than the "
             "trials as a whole, and the interval keeps small noisy groups "
             "from deciding.", "",
             "## Decision per candidate", "",
             "| Candidate | Mitigation | PR-AUC | Largest gating gap "
             "[interval] | Status | Details |", "|---|---|---|---|---|---|"]
    for name, s in summary.items():
        where = (f"{f3(s['max_gap'])} [{f3(s['gap_ci'][0])}, "
                 f"{f3(s['gap_ci'][1])}] ({s['slice']} {s['metric']})"
                 if s["slice"] else "n/a")
        notes = s["blocks"] + s["alerts"]
        if s["unchecked_slices"]:
            notes.append("not checked (too few qualified groups): "
                         + ", ".join(s["unchecked_slices"]))
        details = "; ".join(notes) or "-"
        lines.append(f"| {name} | {s['mitigation']} | {f3(s['pr_auc'])} | "
                     f"{where} | **{s['status']}** | {details} |")

    lines += ["", "## Gaps per slice", "",
              "Cell: gap [interval] (worst group). Status uses the largest "
              "lower bound over the qualified groups.", ""]
    for name in names:
        role = "gating" if name in gating else "report only"
        lines += [f"### {name} ({role})", "",
                  "| Candidate | Qualified groups | "
                  + " | ".join(f"{m} gap" for m in b["metrics"])
                  + " | Status |", "|---|---|" + "---|" * len(b["metrics"])
                  + "---|"]
        for cand in summary:
            sl = slices[(slices["model"] == cand) & (slices["slice"] == name)]
            if sl.empty:
                continue
            cells = [gap_cell(next(sl[sl["metric"] == m].itertuples()))
                     for m in b["metrics"] if (sl["metric"] == m).any()]
            order = ["block", "alert", "ok", "insufficient data"]
            status = min(sl["status"], key=order.index)
            lines.append(f"| {cand} | {int(sl['n_qualified'].max())} | "
                         + " | ".join(cells) + f" | {status} |")
        lines.append("")

    lines += ["## Per-group metrics before and after mitigation "
              "(gating slices)", ""]
    families = {}
    for c in candidates:
        families.setdefault(model_entry(c), []).append(c["name"])
    for name in gating:
        for family, members in families.items():
            g = groups[(groups["slice"] == name)
                       & groups["model"].isin(members)]
            if g.empty:
                continue
            pivot = g.set_index(["group", "n_trials", "n_stopped",
                                 "qualified", "model", "metric"])["value"] \
                .unstack(["model", "metric"])
            cols = [(m, k) for m in members for k in b["metrics"]
                    if (m, k) in pivot.columns]
            lines += [f"**{name}, {family}**", "",
                      "| Group | Trials | Stopped | Qualified | "
                      + " | ".join(f"{m} {k}" for m, k in cols) + " |",
                      "|---|---|---|---|" + "---|" * len(cols),
                      f"| overall | {len(val)} | {int(val['label'].sum())} "
                      "| - | " + " | ".join(
                          f3(val_metrics[m].get(k)) for m, k in cols)
                      + " |"]
            for idx, row in pivot.iterrows():
                lines.append(f"| {idx[0]} | {idx[1]} | {idx[2]} | "
                             f"{'yes' if idx[3] else 'no'} | "
                             + " | ".join(f3(row[c]) for c in cols) + " |")
            lines.append("")

    lines += ["## Mitigation and trade-offs", "",
              "| Candidate | Compared with | PR-AUC | Recall | Precision | "
              "Trials flagged | Largest gating gap |",
              "|---|---|---|---|---|---|---|"]
    for c in candidates:
        if c["mitigation"] == "none":
            continue
        base = c.get("base", model_entry(c))
        if base not in summary or c["name"] not in summary:
            continue
        cells = []
        for k in ("pr_auc", "recall", "precision", "flagged_rate"):
            old, new = val_metrics[base].get(k), val_metrics[c["name"]].get(k)
            cells.append(f"{f3(old)} → {f3(new)}" if None not in (old, new)
                         else "n/a")
        old, new = summary[base]["max_gap"], summary[c["name"]]["max_gap"]
        lines.append(f"| {c['name']} | {base} | " + " | ".join(cells)
                     + f" | {f3(old)} → {f3(new)} |")
    sets = ", ".join(" then ".join(s) for s in threshold_sets(b)) or "none"
    lines += ["",
              "- **reweighted**: trained with the data pipeline's trial "
              "weights (groups under 1% of trials weighted up). Cheap, but "
              "it only moves groups that were boosted, so gaps between large "
              "groups barely change.",
              f"- **group_thresholds** ({sets}): one decision threshold per "
              "group so every group reaches the model's overall recall "
              "(equal opportunity). Cuts are fitted on the validation set "
              f"with {b.get('threshold_folds', 2)}-fold cross-fitting (each "
              "trial scored with cuts fitted without it); the test set and "
              "deployment use cuts fitted on the whole validation set. "
              "Groups with fewer than "
              f"{b['min_stopped']} stopped trials in the fitting part keep "
              "the global threshold.",
              "- **Trade-off**: thresholds change who is flagged, not the "
              "ranking inside a group, so they close recall gaps but not "
              "per-group PR-AUC gaps; mixing groups with different cuts "
              "usually costs some overall PR-AUC and precision, and more "
              "trials get flagged in the groups whose cut is lowered. The "
              "select stage weighs this: score = PR-AUC - "
              f"{gate['bias_penalty']} x largest gating gap.",
              "- Re-sampling is not a separate candidate: with the same "
              "weights it duplicates trials of the boosted groups, which "
              "reweighting already does without copying data.", "",
              "## Charts", ""]
    for family in families:
        for m in b["metrics"]:
            lines.append(f"- `gap_{m}_{family}.png`: {m} gap per slice, "
                         f"{family} before and after mitigation, with "
                         "intervals and the alert and block levels.")
    lines += ["", "## Limits", "",
              "- PR-AUC depends on a group's stopped rate, so part of a "
              "PR-AUC gap reflects how often trials in that group stop, "
              "not only how well the model ranks them.",
              "- Every qualified group is tested, with no correction for "
              "testing many groups; keeping only slices with few, large "
              "groups as gating slices keeps false blocks rare.",
              "- Groups below the size limits are listed in "
              "`model_group_gaps.csv` (qualified = False) but do not count."]
    return "\n".join(lines) + "\n"


def run(ctx):
    b, gate = ctx.params["bias"], ctx.params["gate"]
    threshold = ctx.params["evaluation"]["threshold"]
    seed = int(ctx.params["seed"])
    val = data.load_split(ctx, "val")

    # a rerun of this stage starts again from the trained candidates
    candidates = [c for c in ctx.read_json("candidates.json")
                  if c["mitigation"] != "group_thresholds"]
    scores = {c["name"]: score_candidate(c, load_model(ctx, c), val,
                                         data.load_columns(ctx, c),
                                         threshold)
              for c in candidates}
    extra, oof = add_threshold_candidates(b, candidates, scores, val,
                                          threshold, seed)
    candidates += extra
    scores.update(oof)
    ctx.write_json("candidates.json", candidates)

    ev = ctx.params["evaluation"]
    val_metrics = ctx.read_json("reports/val_metrics.json")
    for name, s in scores.items():
        if name in oof or name not in val_metrics:
            val_metrics[name] = metrics.compute(
                val["label"], s, threshold, ev["top_k_share"])
    ctx.write_json("reports/val_metrics.json", val_metrics)

    table, groups, slices, decisions = assess_split(ctx, val, scores)
    by_name = {c["name"]: c for c in candidates}
    summary = {name: {**d, "family": by_name[name]["family"],
                      "mitigation": by_name[name]["mitigation"],
                      "pr_auc": val_metrics[name]["pr_auc"],
                      "flagged": d["status"] != "ok"}
               for name, d in decisions.items()}

    out = ctx.path("reports", "bias", ".keep").parent
    table.to_csv(out / "model_slices.csv", index=False)
    groups.to_csv(out / "model_group_gaps.csv", index=False)
    slices.to_csv(out / "model_gaps.csv", index=False)
    pd.DataFrame({"nct_id": val["nct_id"], "label": val["label"],
                  **scores}).to_csv(out / "val_scores.csv", index=False)
    ctx.write_json("reports/bias/bias_summary.json", summary)
    (out / "bias_report.md").write_text(
        markdown(ctx, summary, slices, groups, val_metrics, candidates, val))
    families = {}
    for c in candidates:
        families.setdefault(model_entry(c), []).append(c["name"])
    for family, members in families.items():
        for m in b["metrics"]:
            plots.bias_gap_bars(
                slices, m, members, out / f"gap_{m}_{family}.png",
                f"{m} gap per slice, {family} (validation set)",
                gate["alert_bias_gap"], gate["block_bias_gap"])
    tracking.log_artifacts(ctx, out, "bias")

    flagged = []
    for name, s in summary.items():
        tracking.log_metrics(ctx, {"max_bias_gap": s["max_gap"],
                                   "bias_blocked": float(s["blocked"])},
                             prefix=f"val_{name}_")
        if s["status"] != "ok":
            text = "; ".join(s["blocks"] + s["alerts"]) or \
                "no gating slice has enough qualified groups"
            logger.warning("%s: bias %s (%s)", name, s["status"], text)
            flagged.append(f"{name}: {s['status']} ({text})")
    if any(s["blocked"] or s["status"] == "insufficient data"
           for s in summary.values()):
        notify.alert(ctx, "bias check: candidates blocked or without "
                     "enough data",
                     "\n".join(flagged) + "\n\nSee reports/bias/"
                     "bias_report.md.")
    return summary
