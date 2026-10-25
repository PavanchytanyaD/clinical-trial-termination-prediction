"""Data-level bias report: how big each group is and how often its trials stop.

For every slice and group it counts finished trials (label known), stopped
trials and unfinished trials, then flags groups that are too small or whose
stopped rate is far from the overall rate.
"""

import pandas as pd


def explode_slice(trials, slice_name):
    """One row per (trial, group). A trial in "a|b" counts once in each."""
    out = trials.copy()
    out[slice_name] = out[slice_name].str.split("|")
    return out.explode(slice_name)


def slice_table(trials, slice_name, n_finished_total, overall_rate):
    """Counts and stopped rate for each group of one slice."""
    rows = explode_slice(trials, slice_name)
    label = rows.groupby(slice_name)["label"]
    table = pd.DataFrame({
        "n_finished": label.count(),          # count() skips NaN labels
        "n_stopped": label.sum().astype(int),
        "n_unfinished": label.size() - label.count(),
    })
    table["share"] = table.n_finished / n_finished_total
    table["stopped_rate"] = table.n_stopped / table.n_finished
    table["rate_gap"] = table.stopped_rate - overall_rate
    table["unfinished_share"] = table.n_unfinished / label.size()
    table = table.reset_index().rename(columns={slice_name: "group"})
    table.insert(0, "slice", slice_name)
    return table


def data_slice_report(trials, slice_names, min_share, max_rate_gap):
    """Return (report table, summary dict) for all slices."""
    finished = trials[trials.label.notna()]
    overall_rate = finished.label.mean()

    report = pd.concat(
        [slice_table(trials, name, len(finished), overall_rate)
         for name in slice_names],
        ignore_index=True)
    report["too_small"] = report.share < min_share
    report["large_gap"] = report.rate_gap.abs() > max_rate_gap

    def rounded(x):
        # NaN (group with no finished trials) is not valid JSON; use null.
        return None if pd.isna(x) else round(float(x), 4)

    flagged = report[report.too_small | report.large_gap]
    summary = {
        "n_trials": int(len(trials)),
        "n_finished": int(len(finished)),
        "n_unfinished": int(len(trials) - len(finished)),
        "overall_stopped_rate": round(float(overall_rate), 4),
        "min_share": min_share,
        "max_rate_gap": max_rate_gap,
        "flagged_groups": [
            {"slice": r.slice, "group": r.group,
             "share": rounded(r.share),
             "stopped_rate": rounded(r.stopped_rate),
             "too_small": bool(r.too_small),
             "large_gap": bool(r.large_gap)}
            for r in flagged.itertuples()
        ],
    }
    return report, summary
