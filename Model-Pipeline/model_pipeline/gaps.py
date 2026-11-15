"""Subgroup gaps with bootstrap intervals, used by the bias and test stages.

Gap of a metric for one group = overall value minus the group's value, so
it is positive when the group is served worse than the trials as a whole.
Only qualified groups count: at least min_group_size trials and min_stopped
stopped ones, and not UNKNOWN. A slice is checked only when it has at least
min_groups qualified groups; its gap is the largest group gap.

Worst group against overall (instead of best minus worst group) because the
best-minus-worst spread grows with the number of groups by chance alone,
and the best group is often a small lucky one. Against overall, the number
answers one question: how much worse off is the worst group than the trials
as a whole.

Every group gap gets a percentile bootstrap interval: the trials are
resampled with replacement `bootstrap` times (seeded, the same resamples for
every candidate and group) and the gap is computed again each time. A slice
is over a level only if the lower bound of at least one group's gap is above
it, so a large gap in a small, noisy group does not block a model on its
own. With bootstrap: 0 the point gap is used instead.
"""

import numpy as np
import pandas as pd

METRICS = ("recall", "pr_auc")


def memberships(values, sep="|"):
    """{group: boolean mask}; a trial in "a|b" belongs to both groups."""
    parts = pd.Series(np.asarray(values, dtype=str)).str.split(sep).explode()
    masks = {}
    for group, idx in parts.groupby(parts).groups.items():
        mask = np.zeros(len(values), dtype=bool)
        mask[np.asarray(idx)] = True
        masks[group] = mask
    return masks


def resample_counts(n, n_boot, seed):
    """(n_boot, n) array: how often each trial is drawn in each resample."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                     for _ in range(n_boot)]).astype(float) \
        if n_boot else np.zeros((0, n))


def recall(y, pred, w):
    """Recall at the threshold for each row of weights w (n_rows, n)."""
    pos = w @ y
    hit = w @ (y * pred)
    return np.divide(hit, pos, out=np.full(len(w), np.nan), where=pos > 0)


def pr_auc(y, s, w):
    """Average precision (as sklearn computes it) for each row of weights.
    y and s must be sorted by score, highest first."""
    tp = np.cumsum(w * y, axis=1)
    fp = np.cumsum(w * (1 - y), axis=1)
    last = np.r_[s[1:] != s[:-1], True]       # one point per distinct score
    tp, fp = tp[:, last], fp[:, last]
    pos, neg = tp[:, -1:], fp[:, -1:]
    flagged = tp + fp
    prec = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    rec = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)
    ap = (np.diff(rec, axis=1, prepend=0) * prec).sum(axis=1)
    return np.where((pos[:, 0] > 0) & (neg[:, 0] > 0), ap, np.nan)


def _metric(name, y, s, pred, w):
    return recall(y, pred, w) if name == "recall" else pr_auc(y, s, w)


def group_gaps(y, s, slices, threshold, b, counts):
    """One row per (slice, group, metric) with the group's value, the overall
    value, the gap and its bootstrap interval.

    y, s: labels and scores; slices: {slice name: group per trial};
    counts: resample counts from resample_counts (same length as y)."""
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    y = np.asarray(y, dtype=float)[order]
    s = np.asarray(s, dtype=float)[order]
    pred = (s >= threshold).astype(float)
    w_all = np.vstack([np.ones(len(y)), counts[:, order]])
    level = float(b.get("ci_level", 0.9))
    lo_q, hi_q = 50 * (1 - level), 50 * (1 + level)
    metrics = [m for m in b["metrics"] if m in METRICS]
    overall = {m: _metric(m, y, s, pred, w_all) for m in metrics}

    rows = []
    for name, values in slices.items():
        for group, mask in memberships(np.asarray(values)[order]).items():
            n_trials, n_stopped = int(mask.sum()), int(y[mask].sum())
            qualified = (group != "UNKNOWN"
                         and n_trials >= b["min_group_size"]
                         and n_stopped >= b["min_stopped"])
            for m in metrics:
                vals = _metric(m, y[mask], s[mask], pred[mask],
                               w_all[:, mask])
                gap = overall[m] - vals
                boot = gap[1:]
                ok = boot[~np.isnan(boot)]
                ci_low, ci_high = ((np.percentile(ok, lo_q),
                                    np.percentile(ok, hi_q)) if len(ok)
                                   else (gap[0], gap[0]))
                rows.append({"slice": name, "group": group,
                             "n_trials": n_trials, "n_stopped": n_stopped,
                             "qualified": bool(qualified), "metric": m,
                             "overall": overall[m][0], "value": vals[0],
                             "gap": gap[0], "ci_low": float(ci_low),
                             "ci_high": float(ci_high)})
    return pd.DataFrame(rows, columns=[
        "slice", "group", "n_trials", "n_stopped", "qualified", "metric",
        "overall", "value", "gap", "ci_low", "ci_high"])


def level_status(low, alert, block):
    if pd.notna(low) and low > block:
        return "block"
    if pd.notna(low) and low > alert:
        return "alert"
    return "ok"


def slice_gaps(groups, b, gate, gating):
    """One row per (slice, metric): qualified groups, the worst group and
    its gap with interval, the largest lower bound over the qualified groups
    and the status (block / alert / ok / insufficient data). Report slices
    are never above "alert"."""
    rows = []
    for (name, m), g in groups.groupby(["slice", "metric"], sort=False):
        q = g[g["qualified"] & g["gap"].notna()]
        row = {"slice": name, "metric": m, "role": "gate"
               if name in gating else "report", "n_groups": len(g),
               "n_qualified": len(q), "worst_group": None,
               "gap": np.nan, "ci_low": np.nan, "ci_high": np.nan,
               "max_ci_low": np.nan, "max_ci_low_group": None,
               "status": "insufficient data"}
        if len(q) >= int(b.get("min_groups", 2)):
            worst = q.loc[q["gap"].idxmax()]
            strong = q.loc[q["ci_low"].idxmax()]
            status = level_status(strong["ci_low"], gate["alert_bias_gap"],
                                  gate["block_bias_gap"])
            if status == "block" and name not in gating:
                status = "alert"
            row.update(worst_group=worst["group"], gap=worst["gap"],
                       ci_low=worst["ci_low"], ci_high=worst["ci_high"],
                       max_ci_low=strong["ci_low"],
                       max_ci_low_group=strong["group"], status=status)
        rows.append(row)
    return pd.DataFrame(rows)


def decide(slices, gate, gating):
    """Bias decision of one candidate from its slice_gaps rows."""
    gate_rows = slices[slices["slice"].isin(gating)]
    checked = gate_rows[gate_rows["status"] != "insufficient data"]
    unchecked = sorted(set(gate_rows["slice"]) - set(checked["slice"]))
    out = {"max_gap": 0.0, "slice": None, "metric": None,
           "gap_ci": [None, None],
           "checked_slices": sorted(set(checked["slice"])),
           "unchecked_slices": unchecked, "blocks": [], "alerts": []}
    if len(checked):
        top = checked.loc[checked["gap"].idxmax()]
        out.update(max_gap=max(float(top["gap"]), 0.0), slice=top["slice"],
                   metric=top["metric"],
                   gap_ci=[float(top["ci_low"]), float(top["ci_high"])])
    for r in slices.itertuples():
        text = (f"{r.slice} {r.metric}: {r.max_ci_low_group} gap lower "
                f"bound {r.max_ci_low:.3f}")
        if r.status == "block":
            out["blocks"].append(text)
        elif r.status == "alert":
            out["alerts"].append(text)

    if not len(checked):
        out["status"] = "insufficient data"
        out["blocked"] = gate.get("insufficient_bias_data",
                                  "block") == "block"
    else:
        out["status"] = ("block" if out["blocks"] else
                         "alert" if out["alerts"] else "ok")
        out["blocked"] = bool(out["blocks"])
    return out


def assess(rows, scores, slice_names, gating, threshold, b, gate, seed):
    """Gaps of every candidate on one split.

    rows: the split with label and slice columns; scores: {name: scores}.
    Returns (group table, slice table, {name: decision})."""
    counts = resample_counts(len(rows), int(b.get("bootstrap", 200)), seed)
    slices = {n: rows[n].to_numpy() for n in slice_names}
    group_tables, slice_tables, decisions = [], [], {}
    for name, s in scores.items():
        g = group_gaps(rows["label"].to_numpy(), s, slices, threshold, b,
                       counts)
        sl = slice_gaps(g, b, gate, gating)
        decisions[name] = decide(sl, gate, gating)
        group_tables.append(g.assign(model=name))
        slice_tables.append(sl.assign(model=name))
    return (pd.concat(group_tables, ignore_index=True),
            pd.concat(slice_tables, ignore_index=True), decisions)
