"""Stage lead_time: how early the selected model warns, and where it errs.

The model scores a trial when it is registered, so for a stopped trial it
flags (a true positive on the test set) the warning comes
    lead time = stop date - registration date
ahead of the stop. The stop date comes from the AACT studies table (see
data.stop_dates); it is never a feature. Lead times use only trials with
an actual (primary) completion date. For the others (mostly withdrawn and
suspended trials) the only date is the record's last update, which can be
years after the stop, so they are counted but not measured. Trials whose
stop date is before their registration (registered after the fact) are
counted separately.

Error analysis on the same test set:
  - recall per final status (terminated, withdrawn, suspended);
  - recall per time from registration to stop, to see whether the model
    finds early stops as well as late ones;
  - the completed trials with the highest scores (false positives) and the
    stopped trials with the lowest (false negatives), for manual review.
Outputs in reports/lead_time/.
"""

import logging

import numpy as np
import pandas as pd

from model_pipeline import plots, tracking

logger = logging.getLogger(__name__)
DAYS_PER_MONTH = 30.44
ACTUAL_DATES = ("completion", "primary_completion")
BUCKETS = [0, 6, 12, 24, 36, np.inf]
SHOW = ["nct_id", "score", "probability", "overall_status", "phase",
        "sponsor_class", "therapeutic_area", "study_first_submitted_date",
        "stop_date"]


def months_to_stop(rows):
    """Months from registration to an actual stop date; NaN without one."""
    start = pd.to_datetime(rows["study_first_submitted_date"],
                           errors="coerce")
    months = (pd.to_datetime(rows["stop_date"]) - start).dt.days \
        / DAYS_PER_MONTH
    return months.where(rows["stop_date_source"].isin(ACTUAL_DATES))


def summary(months):
    m = months.dropna()
    ahead = m[m > 0]
    if ahead.empty:
        return {"n": int(len(m)), "n_after_the_fact": int((m <= 0).sum())}
    return {"n": int(len(m)),
            "n_after_the_fact": int((m <= 0).sum()),
            "median_months": float(ahead.median()),
            "q25_months": float(ahead.quantile(0.25)),
            "q75_months": float(ahead.quantile(0.75)),
            "share_over_12_months": float((ahead > 12).mean())}


def recall_by(stopped, column):
    g = stopped.groupby(column, observed=True)["flagged"]
    return pd.DataFrame({"stopped_trials": g.size(),
                         "flagged": g.sum(),
                         "recall": g.mean()}).reset_index()


def table_md(df):
    fmt = (lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def markdown(result, by_status, by_time):
    tp, every = result["true_positives"], result["all_stopped"]
    lines = ["# Lead time and error analysis (test set)", "",
             f"Model: {result['candidate']}. Lead time = stop date - "
             "registration date, for stopped trials the model flagged.", ""]
    if "median_months" in tp:
        lines += [f"- Flagged stopped trials: {result['n_flagged']} of "
                  f"{result['n_stopped']}; {tp['n']} with an actual stop "
                  f"date are measured ({tp['n_after_the_fact']} of them "
                  "registered after they stopped, left out below)",
                  f"- Median lead time: **{tp['median_months']:.1f} months** "
                  f"(IQR {tp['q25_months']:.1f} to {tp['q75_months']:.1f})",
                  f"- Flagged more than 12 months ahead: "
                  f"**{100 * tp['share_over_12_months']:.0f}%**",
                  f"- All stopped trials, flagged or not: median "
                  f"{every.get('median_months', float('nan')):.1f} months "
                  "from registration to stop"]
    lines += ["", "## Recall per final status", "",
              table_md(by_status), "",
              "## Recall per time from registration to stop", "",
              table_md(by_time), "",
              "Only trials with an actual stop date; withdrawn and "
              "suspended trials often have none.", "",
              "Top false positives and false negatives: "
              "top_false_positives.csv, top_false_negatives.csv."]
    return "\n".join(lines) + "\n"


def run(ctx):
    out = ctx.path("reports", "lead_time", ".keep").parent
    rows = pd.read_pickle(ctx.path("data", "test_scores.pkl"))
    name = ctx.read_json("reports/selection.json")["selected"]["name"]
    if "stop_date" not in rows.columns:
        logger.warning("no stop dates in this data version; lead time "
                       "skipped (add the AACT studies table to data.files)")
        (out / "lead_time.md").write_text("No stop dates in this data "
                                          "version; lead time skipped.\n")
        return None

    rows = rows.assign(months_to_stop=months_to_stop(rows))
    stopped = rows[rows["label"] == 1].copy()
    tp = stopped[stopped["flagged"]]
    result = {"candidate": name,
              "n_stopped": int(len(stopped)), "n_flagged": int(len(tp)),
              "true_positives": summary(tp["months_to_stop"]),
              "all_stopped": summary(stopped["months_to_stop"]),
              "missed_stopped": summary(
                  stopped.loc[~stopped["flagged"], "months_to_stop"]),
              "stop_date_sources": stopped["stop_date_source"]
              .value_counts().to_dict()}

    status = "overall_status" if "overall_status" in stopped else "label"
    by_status = recall_by(stopped, status)
    stopped["time_to_stop"] = pd.cut(
        stopped["months_to_stop"], BUCKETS, right=False,
        labels=["0-6 months", "6-12", "12-24", "24-36", "36+"])
    by_time = recall_by(stopped.dropna(subset=["time_to_stop"]),
                        "time_to_stop")
    show = [c for c in SHOW if c in rows.columns]
    rows[rows["label"] == 0].nlargest(25, "score")[show].to_csv(
        out / "top_false_positives.csv", index=False)
    stopped.nsmallest(25, "score")[show].to_csv(
        out / "top_false_negatives.csv", index=False)

    ctx.write_json("reports/lead_time/lead_time.json", result)
    by_status.to_csv(out / "recall_by_status.csv", index=False)
    by_time.to_csv(out / "recall_by_time_to_stop.csv", index=False)
    (out / "lead_time.md").write_text(markdown(result, by_status, by_time))
    ahead = tp["months_to_stop"][tp["months_to_stop"] > 0]
    if len(ahead):
        plots.histogram(ahead, out / "lead_time_hist.png",
                        f"Lead time of flagged stopped trials ({name})",
                        "months from registration to stop")
    tracking.log_metrics(ctx, {f"lead_time_{k}": v for k, v in
                               result["true_positives"].items()})
    tracking.log_artifacts(ctx, out, "lead_time")
    logger.info("lead time: %s", result["true_positives"])
    return result
