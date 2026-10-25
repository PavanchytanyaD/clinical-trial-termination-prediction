"""Pipeline step 4, validate a snapshot against the baseline schema and
statistics, so data quality is checked over time.

Two checks:

1. Schema: the Great Expectations baseline suite (schema/baseline/) runs on
   data/processed/structured_features.csv. Before a baseline is frozen, the
   current snapshot's own suite is used and the report says so.
2. Statistics drift: the snapshot's statistics are compared with the
   baseline statistics: row count, stopped-trial rate, missing rate per
   column and mean shift per numeric column (in baseline standard
   deviations). Drift findings are warnings.

Writes reports/validation/validation.json (every check), summary.json
(counts, tracked by DVC as metrics) and validation.md. Emails an alert when
something fails (validate.alert_on) and exits with code 1 when the pipeline
must stop (validate.fail_on: critical, warning or never), so Airflow marks
the task failed and later steps do not run.

    python -m scripts.validate
"""

import datetime as dt
import json
import logging
import os

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
import great_expectations as gx  # noqa: E402
from great_expectations.data_context.types.base import ProgressBarsConfig  # noqa: E402
from great_expectations.expectations.expectation_configuration import (  # noqa: E402
    ExpectationConfiguration)

from scripts.alerts import send_alert  # noqa: E402
from scripts.config import (BASELINE_STATS, BASELINE_SUITE, GENERATED_SUITE,  # noqa: E402
                            STATS_JSON, VALIDATION_JSON, VALIDATION_MD,
                            VALIDATION_SUMMARY, load_params)
from scripts.schema_stats import load_features  # noqa: E402
from scripts.tracking import tracked_step  # noqa: E402

logger = logging.getLogger("validate")

DEFAULTS = {"fail_on": "critical", "alert_on": "warning",
            "max_row_change": 0.5, "max_stopped_rate_change": 0.05,
            "max_missing_increase": 0.05, "max_mean_shift_std": 0.5}
SEVERITY_RANK = {"never": 3, "critical": 2, "warning": 1}


class ValidationFailed(Exception):
    """The snapshot failed checks at or above validate.fail_on."""

    alerted = True  # the validation alert already went out


def load_suite(path):
    d = json.loads(path.read_text())
    expectations = [
        ExpectationConfiguration(type=e["type"], kwargs=e["kwargs"],
                                 meta=e.get("meta") or {},
                                 severity=e.get("severity", "critical")
                                 ).to_domain_obj()
        for e in d["expectations"]]
    return gx.ExpectationSuite(name=d["name"], expectations=expectations,
                               meta=d.get("meta") or {})


def run_suite(df, suite):
    """Run the suite on df; one dict per expectation.

    Expectations on a column the data no longer has are not run (they would
    only error); each missing column is reported once as a critical failure,
    next to the table-level column-set check.
    """
    missing = sorted({e.column for e in suite.expectations
                      if getattr(e, "column", None) and e.column not in df})
    rows = [{"check": "schema", "expectation": "column_present",
             "column": col, "severity": "critical", "success": False,
             "observed": "column missing from the data"} for col in missing]
    suite.expectations = [e for e in suite.expectations
                          if getattr(e, "column", None) not in missing]
    ctx = gx.get_context(mode="ephemeral")
    ctx.variables.progress_bars = ProgressBarsConfig(globally=False)
    batch = (ctx.data_sources.add_pandas("features")
             .add_dataframe_asset("structured_features")
             .add_batch_definition_whole_dataframe("snapshot"))
    suite = ctx.suites.add(suite)
    result = gx.ValidationDefinition(data=batch, suite=suite,
                                     name="snapshot_validation").run(
        batch_parameters={"dataframe": df}, result_format="SUMMARY")
    for r in result.to_json_dict()["results"]:
        cfg, res = r["expectation_config"], r["result"] or {}
        exception = (r.get("exception_info") or {}).get("raised_exception")
        observed = res.get("observed_value")
        mismatched = (res.get("details") or {}).get("mismatched")
        if mismatched:  # column-set check: show only what differs
            observed = {"unexpected_columns": mismatched.get("unexpected"),
                        "missing_columns": mismatched.get("missing")}
        rows.append({
            "check": "schema",
            "expectation": cfg["type"],
            "column": cfg["kwargs"].get("column"),
            "severity": cfg.get("severity", "critical"),
            "success": bool(r["success"]) and not exception,
            "observed": observed,
            "unexpected_count": res.get("unexpected_count"),
            "unexpected_percent": res.get("unexpected_percent"),
            "examples": (res.get("partial_unexpected_list") or [])[:5],
            "kwargs": {k: v for k, v in cfg["kwargs"].items()
                       if k not in ("column", "batch_id")},
            "error": (r.get("exception_info") or {}).get("exception_message")
            if exception else None,
        })
    return rows


def _finding(name, column, baseline, current, limit, ok):
    return {"check": "drift", "expectation": name, "column": column,
            "severity": "warning", "success": bool(ok),
            "observed": {"baseline": baseline, "current": current,
                         "limit": limit}}


def compare_statistics(current, baseline, params=None):
    """Drift findings between two statistics.json files (all warnings)."""
    p = {**DEFAULTS, **(params or {})}
    out = []
    cd, bd = current["dataset"], baseline["dataset"]
    if bd["n_rows"]:
        change = abs(cd["n_rows"] - bd["n_rows"]) / bd["n_rows"]
        out.append(_finding("row_count_change", None, bd["n_rows"],
                            cd["n_rows"], p["max_row_change"],
                            change <= p["max_row_change"]))
    if None not in (cd.get("stopped_rate"), bd.get("stopped_rate")):
        diff = abs(cd["stopped_rate"] - bd["stopped_rate"])
        out.append(_finding("stopped_rate_change", "label", bd["stopped_rate"],
                            cd["stopped_rate"], p["max_stopped_rate_change"],
                            diff <= p["max_stopped_rate_change"]))
    for name, b in baseline["columns"].items():
        c = current["columns"].get(name)
        if c is None:
            continue  # a dropped column is a critical schema failure already
        inc = (c["missing_rate"] or 0) - (b["missing_rate"] or 0)
        if inc > 0:
            out.append(_finding("missing_rate_increase", name,
                                b["missing_rate"], c["missing_rate"],
                                p["max_missing_increase"],
                                inc <= p["max_missing_increase"]))
        numeric = b["kind"] == "numeric" and b.get("std")
        if numeric and c.get("mean") is not None:
            shift = abs(c["mean"] - b["mean"]) / b["std"]
            out.append(_finding("mean_shift_std", name, b["mean"], c["mean"],
                                p["max_mean_shift_std"],
                                shift <= p["max_mean_shift_std"]))
    return out


def summarize(checks, suite_source, snapshot, fail_on):
    failed = [c for c in checks if not c["success"]]
    n_critical = sum(c["severity"] == "critical" for c in failed)
    n_warning = sum(c["severity"] == "warning" for c in failed)
    worst = "critical" if n_critical else ("warning" if n_warning else None)
    stop = worst is not None and SEVERITY_RANK[worst] >= SEVERITY_RANK[fail_on]
    return {
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "snapshot": snapshot,
        "suite": suite_source,
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed_critical": n_critical,
        "failed_warning": n_warning,
        "success": n_critical == 0,
        "stop_pipeline": stop,
    }


def describe(check):
    col = f" on {check['column']}" if check["column"] else ""
    obs = check.get("observed")
    detail = ""
    if check.get("unexpected_percent") is not None:
        detail = (f": {check['unexpected_count']} unexpected "
                  f"({check['unexpected_percent']:.2f}%)")
        if check.get("examples"):
            detail += f", e.g. {check['examples'][:3]}"
    elif obs is not None:
        detail = f": observed {obs}"
    if check.get("error"):
        detail += f" (error: {check['error']})"
    return f"[{check['severity']}] {check['expectation']}{col}{detail}"


def write_reports(summary, checks):
    VALIDATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(
        {"summary": summary, "checks": checks}, indent=2, default=str))
    VALIDATION_SUMMARY.write_text(json.dumps(summary, indent=2))
    failed = [c for c in checks if not c["success"]]
    lines = [f"# Validation: snapshot {summary['snapshot']}", "",
             f"- Suite: {summary['suite']}",
             f"- Checks: {summary['checks']}, passed: {summary['passed']}, "
             f"critical failures: {summary['failed_critical']}, warnings: "
             f"{summary['failed_warning']}",
             f"- Pipeline stopped: {summary['stop_pipeline']}", ""]
    lines += (["## Failed checks", ""] + [f"- {describe(c)}" for c in failed]
              if failed else ["All checks passed."])
    VALIDATION_MD.write_text("\n".join(lines) + "\n")


def validate(params=None):
    p = {**DEFAULTS, **(params or {})}
    if p["fail_on"] not in SEVERITY_RANK or p["alert_on"] not in SEVERITY_RANK:
        raise ValueError("validate.fail_on and validate.alert_on must be "
                         "critical, warning or never")
    current = json.loads(STATS_JSON.read_text())
    if BASELINE_SUITE.exists():
        suite_path, suite_source = BASELINE_SUITE, "baseline"
    else:
        suite_path, suite_source = GENERATED_SUITE, "current snapshot " \
            "(no baseline frozen yet)"
        logger.warning("no baseline in schema/baseline/: validating against "
                       "this snapshot's own schema")
    checks = run_suite(load_features(), load_suite(suite_path))
    checks.sort(key=lambda c: c["expectation"] != "column_present")
    if BASELINE_STATS.exists():
        checks += compare_statistics(current,
                                     json.loads(BASELINE_STATS.read_text()), p)
    summary = summarize(checks, suite_source, current.get("snapshot"),
                        p["fail_on"])
    write_reports(summary, checks)
    return summary, checks


def main():
    with tracked_step("validate") as run:
        params = {**DEFAULTS, **load_params().get("validate", {})}
        summary, checks = validate(params)
        run["metrics"].update({k: v for k, v in summary.items()
                               if isinstance(v, (int, float, bool))})
        failed = [c for c in checks if not c["success"]]
        for c in failed:
            (logger.error if c["severity"] == "critical"
             else logger.warning)(describe(c))
        logger.info("%d/%d checks passed (%d critical, %d warning failures)",
                    summary["passed"], summary["checks"],
                    summary["failed_critical"], summary["failed_warning"])
        worst = ("critical" if summary["failed_critical"] else
                 "warning" if summary["failed_warning"] else None)
        if worst and SEVERITY_RANK[worst] >= SEVERITY_RANK[params["alert_on"]]:
            body = "\n".join(
                [f"Snapshot: {summary['snapshot']}",
                 f"Run: {run['run_id']}",
                 f"Pipeline stopped: {summary['stop_pipeline']}", ""]
                + [describe(c) for c in failed[:50]]
                + ([f"... and {len(failed) - 50} more"]
                   if len(failed) > 50 else [])
                + ["", f"Full report: {VALIDATION_MD}"])
            send_alert(f"validation {worst}: {len(failed)} checks failed "
                       f"(snapshot {summary['snapshot']})", body)
        if summary["stop_pipeline"]:
            raise ValidationFailed(
                f"{summary['failed_critical']} critical and "
                f"{summary['failed_warning']} warning checks failed; see "
                f"{VALIDATION_MD}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
