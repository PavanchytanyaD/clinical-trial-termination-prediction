"""Pipeline stage: anomaly detection and alerts (rubric item 8).

Checks the snapshot's feature table for values that are wrong or suspicious
on their own, without needing a baseline (scripts/validate.py compares with
the frozen baseline schema and statistics instead):

critical (the data is broken; the run stops before publish)
  - duplicate trial ids, or ids that are not NCT + 8 digits
  - labels other than 1 (stopped), 0 (completed) or empty (still running)
  - impossible values: negative counts, ages outside 0-120 years, a minimum
    age above the maximum age, start or registration years outside
    1900 to next year
  - no finished trial of one class (all stopped or all completed)
warning (the data looks unusual; someone should look)
  - missing values: share of trials with a column's _missing flag above
    anomalies.max_missing_rate
  - outliers: share of values outside [Q1 - k IQR, Q3 + k IQR] above
    anomalies.max_outlier_share, for every non-binary numeric column
  - dates that are not YYYY-MM or YYYY-MM-DD

Writes reports/anomalies/anomalies.json (every finding with counts and
examples), anomalies.md and summary.json (DVC metrics). Emails an alert when
something is found (anomalies.alert_on) and exits with code 1 when the run
must stop (anomalies.fail_on).

    python -m scripts.anomalies
"""

import datetime as dt
import json
import logging


from scripts.alerts import send_alert
from scripts.config import (ANOMALY_JSON, ANOMALY_MD, ANOMALY_SUMMARY,
                            FEATURES_CSV, load_params)
from scripts.schema_stats import (DATE_COLUMNS, DATE_REGEX, ID_COLUMN,
                                  LABEL_COLUMN, NCT_REGEX, column_kind,
                                  iqr_outliers, load_features, real_values)
from scripts.tracking import tracked_step
from src.error_handling import DataValidationError

logger = logging.getLogger("anomalies")

DEFAULTS = {"fail_on": "critical", "alert_on": "warning",
            "max_missing_rate": 0.3, "max_outlier_share": 0.05,
            "iqr_k": 3.0, "max_invalid_date_share": 0.01}
SEVERITY_RANK = {"never": 3, "critical": 2, "warning": 1}
AGE_COLUMNS = ["min_age_years", "max_age_years"]
YEAR_COLUMNS = ["start_year", "registration_year"]


class AnomaliesFound(DataValidationError):
    """Critical anomalies at or above anomalies.fail_on."""

    alerted = True  # the anomaly alert already went out


def finding(check, severity, column=None, count=0, share=None, examples=(),
            detail=""):
    return {"check": check, "severity": severity, "column": column,
            "count": int(count),
            "share": None if share is None else round(float(share), 4),
            "examples": [str(x) for x in list(examples)[:5]],
            "detail": detail}


def present(df, col):
    """Rows where col has a real value (its _missing flag is not set)."""
    flag = f"{col}_missing"
    mask = df[col].notna()
    if flag in df:
        mask &= df[flag] == 0
    return mask


def id_anomalies(df):
    out = []
    ids = df[ID_COLUMN].astype(str)
    dupes = ids[ids.duplicated(keep=False)]
    if len(dupes):
        out.append(finding("duplicate_ids", "critical", ID_COLUMN,
                           dupes.nunique(), dupes.nunique() / len(df),
                           dupes.unique()))
    bad = ids[~ids.str.match(NCT_REGEX)]
    if len(bad):
        out.append(finding("invalid_ids", "critical", ID_COLUMN, len(bad),
                           len(bad) / len(df), bad))
    return out


def label_anomalies(df):
    out = []
    label = df[LABEL_COLUMN]
    bad = label[label.notna() & ~label.isin([0, 1])]
    if len(bad):
        out.append(finding("invalid_labels", "critical", LABEL_COLUMN,
                           len(bad), len(bad) / len(df), bad))
    finished = label.dropna()
    if len(finished) and finished.nunique() < 2:
        out.append(finding(
            "single_class", "critical", LABEL_COLUMN, len(finished),
            detail="every finished trial has the same label "
                   f"({finished.iloc[0]:g})"))
    return out


def range_anomalies(df, year_max):
    out = []
    checks = []
    for col in df.columns:
        if col.startswith("n_") and not col.endswith("_missing"):
            checks.append((col, 0, None, "negative count"))
    checks += [(c, 0, 120, "age outside 0-120 years") for c in AGE_COLUMNS]
    checks += [(c, 1900, year_max, f"year outside 1900-{year_max}")
               for c in YEAR_COLUMNS]
    for col, low, high, why in checks:
        if col not in df:
            continue
        values = real_values(df, col)
        bad = values[(values < low) | ((values > high) if high else False)]
        if len(bad):
            out.append(finding("impossible_values", "critical", col,
                               len(bad), len(bad) / len(df), bad.unique(),
                               why))
    if set(AGE_COLUMNS) <= set(df.columns):
        both = present(df, AGE_COLUMNS[0]) & present(df, AGE_COLUMNS[1])
        swapped = df[both & (df[AGE_COLUMNS[0]] > df[AGE_COLUMNS[1]])]
        if len(swapped):
            out.append(finding("impossible_values", "critical",
                               "min_age_years", len(swapped),
                               len(swapped) / len(df), swapped[ID_COLUMN],
                               "minimum age above maximum age"))
    return out


def missing_anomalies(df, max_rate):
    out = []
    for col in df.columns:
        if not col.endswith("_missing") or col[:-8] not in df:
            continue
        rate = df[col].mean()
        if rate > max_rate:
            out.append(finding("missing_values", "warning", col[:-8],
                               df[col].sum(), rate,
                               detail=f"missing in {rate:.1%} of trials "
                                      f"(limit {max_rate:.0%})"))
    return out


def outlier_anomalies(df, max_share, k):
    out = []
    for col in df.columns:
        if col in (ID_COLUMN, LABEL_COLUMN) or col.endswith("_missing"):
            continue
        if column_kind(col, df[col]) != "numeric":
            continue
        values = real_values(df, col)
        found = iqr_outliers(values, k)
        if found is None:
            continue
        low, high, outliers = found
        share = len(outliers) / len(values)
        if share > max_share:
            out.append(finding(
                "outliers", "warning", col, len(outliers), share,
                outliers.sort_values(ascending=False).unique(),
                f"outside [{low:.4g}, {high:.4g}] (Q1 - {k:g} IQR, "
                f"Q3 + {k:g} IQR) for {share:.1%} of trials"))
    return out


def date_anomalies(df, max_share):
    out = []
    for col in DATE_COLUMNS:
        if col not in df:
            continue
        values = df[col].dropna().astype(str)
        bad = values[~values.str.match(DATE_REGEX)]
        share = len(bad) / len(df) if len(df) else 0
        if share > max_share:
            out.append(finding("invalid_dates", "warning", col, len(bad),
                               share, bad.unique()))
    return out


def detect(df, params=None, today=None):
    """All anomaly findings for one feature table."""
    p = {**DEFAULTS, **(params or {})}
    year_max = (today or dt.date.today()).year + 1
    return (id_anomalies(df) + label_anomalies(df)
            + range_anomalies(df, year_max)
            + missing_anomalies(df, float(p["max_missing_rate"]))
            + outlier_anomalies(df, float(p["max_outlier_share"]),
                                float(p["iqr_k"]))
            + date_anomalies(df, float(p["max_invalid_date_share"])))


def summarize(findings, n_rows, fail_on):
    n_critical = sum(f["severity"] == "critical" for f in findings)
    n_warning = sum(f["severity"] == "warning" for f in findings)
    worst = "critical" if n_critical else ("warning" if n_warning else None)
    return {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "rows": n_rows,
        "critical": n_critical,
        "warning": n_warning,
        "worst": worst,
        "stop_pipeline": worst is not None
        and SEVERITY_RANK[worst] >= SEVERITY_RANK[fail_on],
    }


def describe(f):
    col = f" in {f['column']}" if f["column"] else ""
    share = f" ({f['share']:.2%})" if f["share"] is not None else ""
    examples = f", e.g. {', '.join(f['examples'][:3])}" if f["examples"] \
        else ""
    detail = f": {f['detail']}" if f["detail"] else ""
    return (f"[{f['severity']}] {f['check']}{col}: {f['count']} "
            f"rows{share}{detail}{examples}")


def write_reports(summary, findings):
    ANOMALY_JSON.parent.mkdir(parents=True, exist_ok=True)
    ANOMALY_JSON.write_text(json.dumps(
        {"summary": summary, "findings": findings}, indent=2))
    ANOMALY_SUMMARY.write_text(json.dumps(summary, indent=2))
    lines = [f"# Anomalies ({summary['rows']} trials)", "",
             f"- Critical: {summary['critical']}, warnings: "
             f"{summary['warning']}",
             f"- Pipeline stopped: {summary['stop_pipeline']}", ""]
    lines += [f"- {describe(f)}" for f in findings] or ["No anomalies."]
    ANOMALY_MD.write_text("\n".join(lines) + "\n")


def main():
    with tracked_step("anomalies") as run:
        params = {**DEFAULTS, **load_params().get("anomalies", {})}
        if params["fail_on"] not in SEVERITY_RANK or \
                params["alert_on"] not in SEVERITY_RANK:
            raise ValueError("anomalies.fail_on and anomalies.alert_on must "
                             "be critical, warning or never")
        df = load_features(FEATURES_CSV)
        findings = detect(df, params)
        summary = summarize(findings, len(df), params["fail_on"])
        write_reports(summary, findings)
        run["metrics"].update({k: summary[k] for k in
                               ("rows", "critical", "warning",
                                "stop_pipeline")})
        for f in findings:
            (logger.error if f["severity"] == "critical"
             else logger.warning)(describe(f))
        logger.info("%d critical, %d warning anomalies in %d trials",
                    summary["critical"], summary["warning"], len(df))
        worst = summary["worst"]
        if worst and SEVERITY_RANK[worst] >= SEVERITY_RANK[params["alert_on"]]:
            send_alert(
                f"anomalies {worst}: {len(findings)} found",
                "\n".join([f"Run: {run['run_id']}",
                           f"Pipeline stopped: {summary['stop_pipeline']}",
                           ""] + [describe(f) for f in findings]
                          + ["", f"Full report: {ANOMALY_MD}"]))
        if summary["stop_pipeline"]:
            raise AnomaliesFound(
                f"{summary['critical']} critical anomalies; see {ANOMALY_MD}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
