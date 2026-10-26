"""Pipeline step 3, data schema and statistics generation.

Reads data/processed/structured_features.csv and writes:

* reports/statistics/statistics.json and statistics.md: dataset statistics
  (rows, label balance) and per-column statistics (type, missing rate,
  distinct values, min / max / mean / std / quantiles, top categories,
  invalid dates or ids),
* schema/generated/expectation_suite.json: a Great Expectations suite
  inferred from this snapshot (the schema): the exact column set, types,
  nulls, value sets, ranges and id / date formats.

Each expectation has a severity. "critical" means the data is broken and the
pipeline must stop (a column added or dropped, duplicate or malformed trial
ids, a label outside {0, 1}, a non-numeric feature, nulls in a feature that
had none). "warning" means the data looks different and a person should
check it (values outside the reference range, a new category, more missing
dates).

The suite generated from a trusted snapshot is frozen as the baseline that
later snapshots are validated against (scripts/validate.py):

    python -m scripts.schema_stats            # pipeline stage
    python -m scripts.schema_stats --freeze   # copy the current suite and
                                              # statistics to schema/baseline/
                                              # (run_stage freeze, in GCS)
"""

import argparse
import datetime as dt
import json
import logging
import os
import shutil

import pandas as pd

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402

from scripts.config import (BASELINE_DIR, BASELINE_INFO, BASELINE_STATS,  # noqa: E402
                            BASELINE_SUITE, FEATURES_CSV, FEATURES_META,
                            GENERATED_SUITE, RAW_DIR, STATS_JSON, STATS_MD,
                            load_params)
from scripts.tracking import fingerprint, git_commit, tracked_step  # noqa: E402

logger = logging.getLogger("schema_stats")

SUITE_NAME = "structured_features"
ID_COLUMN = "nct_id"
LABEL_COLUMN = "label"
STATUS_COLUMN = "overall_status"
DATE_COLUMNS = ["start_date", "study_first_submitted_date"]
NCT_REGEX = r"^NCT\d{8}$"
DATE_REGEX = r"^\d{4}-\d{2}(?:-\d{2})?$"
NUMERIC_TYPES = ["int8", "int16", "int32", "int64", "float32", "float64",
                 "Int64", "Float64"]
COUNT_SUFFIXES = ("_years", "_year", "_words", "_count")
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
DEFAULTS = {"min_rows": 100, "range_slack": 0.5, "mostly": 0.99,
            "max_missing_increase": 0.05, "categorical_max_distinct": 50}


def load_features(path=FEATURES_CSV):
    text_cols = [ID_COLUMN, STATUS_COLUMN] + DATE_COLUMNS
    return pd.read_csv(path, dtype={c: "str" for c in text_cols})


def column_kind(name, series):
    if name == ID_COLUMN:
        return "id"
    if name == LABEL_COLUMN:
        return "label"
    if name in DATE_COLUMNS:
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna().unique()
        # counts (n_*) that happen to be 0/1 in one snapshot stay numeric
        is_count = name.startswith("n_") or name.endswith(COUNT_SUFFIXES)
        if len(values) and set(values) <= {0, 1} and not is_count:
            return "binary"
        return "numeric"
    return "categorical"


def _num(x):
    """JSON-safe float (None for NaN)."""
    return None if x is None or pd.isna(x) else round(float(x), 6)


def column_statistics(name, series):
    kind = column_kind(name, series)
    n = len(series)
    missing = int(series.isna().sum())
    s = {"kind": kind, "dtype": str(series.dtype), "count": n - missing,
         "missing": missing, "missing_rate": _num(missing / n) if n else 0.0,
         "distinct": int(series.nunique(dropna=True))}
    values = series.dropna()
    if kind in ("numeric", "binary", "label"):
        s.update({"min": _num(values.min()), "max": _num(values.max()),
                  "mean": _num(values.mean()), "std": _num(values.std()),
                  "zeros_rate": _num((values == 0).mean())
                  if len(values) else None})
        if kind == "numeric":
            qs = values.quantile(QUANTILES) if len(values) else {}
            s["quantiles"] = {f"p{int(q * 100):02d}": _num(qs[q])
                              for q in QUANTILES} if len(values) else {}
    elif kind == "categorical":
        top = values.astype(str).value_counts().head(20)
        s["top_values"] = {str(k): int(v) for k, v in top.items()}
    elif kind == "date":
        text = values.astype(str)
        s["min"] = text.min() if len(text) else None
        s["max"] = text.max() if len(text) else None
        s["invalid_rate"] = _num((~text.str.match(DATE_REGEX)).mean()) \
            if len(text) else 0.0
    elif kind == "id":
        text = values.astype(str)
        s["duplicates"] = int(text.duplicated().sum())
        s["invalid_rate"] = _num((~text.str.match(NCT_REGEX)).mean()) \
            if len(text) else 0.0
    return s


def snapshot_info():
    info = {}
    manifest = RAW_DIR / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        info = {"snapshot": m.get("snapshot"), "source": m.get("source")}
    if FEATURES_META.exists():
        info["builder"] = json.loads(FEATURES_META.read_text()).get("builder")
    return info


def compute_statistics(df, info=None):
    labels = df[LABEL_COLUMN] if LABEL_COLUMN in df else pd.Series(dtype=float)
    labelled = labels.dropna()
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        **(info or {}),
        "dataset": {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "label_counts": {"stopped": int((labels == 1).sum()),
                             "completed": int((labels == 0).sum()),
                             "unfinished": int(labels.isna().sum())},
            "stopped_rate": _num(labelled.mean()) if len(labelled) else None,
        },
        "columns": {c: column_statistics(c, df[c]) for c in df.columns},
    }


def _upper_bound(max_value, slack):
    return float(max_value + max(abs(max_value) * slack, 1.0))


def infer_suite(stats, params=None):
    """Build the expectation suite (the schema) from snapshot statistics."""
    p = {**DEFAULTS, **(params or {})}
    mostly = float(p["mostly"])
    cols = stats["columns"]
    exp = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=list(cols), exact_match=True, severity="critical",
            meta={"why": "a column was added or dropped upstream"}),
        gxe.ExpectTableRowCountToBeBetween(
            min_value=int(p["min_rows"]), severity="critical",
            meta={"why": "snapshot is empty or truncated"}),
    ]
    for name, c in cols.items():
        kind = c["kind"]
        if kind == "id":
            exp += [
                gxe.ExpectColumnValuesToNotBeNull(column=name,
                                                  severity="critical"),
                gxe.ExpectColumnValuesToBeUnique(column=name,
                                                 severity="critical"),
                gxe.ExpectColumnValuesToMatchRegex(column=name,
                                                   regex=NCT_REGEX,
                                                   severity="critical"),
            ]
        elif kind == "label":
            exp.append(gxe.ExpectColumnValuesToBeInSet(
                column=name, value_set=[0, 1], severity="critical",
                meta={"why": "1 = stopped early, 0 = completed, null = "
                             "still running"}))
        elif kind == "date":
            exp.append(gxe.ExpectColumnValuesToMatchRegex(
                column=name, regex=DATE_REGEX, mostly=mostly,
                severity="warning"))
        elif kind == "categorical":
            exp.append(gxe.ExpectColumnValuesToNotBeNull(
                column=name, severity="critical"))
            if c["distinct"] <= int(p["categorical_max_distinct"]):
                exp.append(gxe.ExpectColumnValuesToBeInSet(
                    column=name, value_set=sorted(c["top_values"]),
                    mostly=mostly, severity="warning",
                    meta={"why": "new category value"}))
        else:  # numeric or binary feature
            exp.append(gxe.ExpectColumnValuesToBeInTypeList(
                column=name, type_list=NUMERIC_TYPES, severity="critical"))
            if c["missing"] == 0:
                exp.append(gxe.ExpectColumnValuesToNotBeNull(
                    column=name, severity="critical",
                    meta={"why": "features are filled during preprocessing"}))
            else:
                allowed = min(1.0, c["missing_rate"]
                              + float(p["max_missing_increase"]))
                exp.append(gxe.ExpectColumnValuesToNotBeNull(
                    column=name, mostly=round(1 - allowed, 6),
                    severity="warning"))
            if kind == "binary":
                exp.append(gxe.ExpectColumnValuesToBeInSet(
                    column=name, value_set=[0, 1], severity="critical",
                    meta={"why": "one-hot and flag columns are 0 or 1"}))
            elif c["min"] is not None:
                low = 0.0 if c["min"] >= 0 else float(c["min"])
                exp.append(gxe.ExpectColumnValuesToBeBetween(
                    column=name, min_value=low,
                    max_value=_upper_bound(c["max"], float(p["range_slack"])),
                    mostly=mostly, severity="warning",
                    meta={"reference_min": c["min"],
                          "reference_max": c["max"]}))
    return gx.ExpectationSuite(
        name=SUITE_NAME, expectations=exp,
        meta={"generated_at": stats.get("generated_at"),
              "reference_snapshot": stats.get("snapshot"),
              "reference_rows": stats["dataset"]["n_rows"],
              "params": p})


def write_markdown(stats, path=STATS_MD):
    d = stats["dataset"]
    lines = [
        f"# Statistics: snapshot {stats.get('snapshot', '?')}", "",
        f"- Generated: {stats['generated_at']}",
        f"- Source: {stats.get('source', '?')}, features built by "
        f"{stats.get('builder', '?')}",
        f"- Rows: {d['n_rows']}, columns: {d['n_columns']}",
        f"- Labels: {d['label_counts']}, stopped rate among finished trials: "
        f"{d['stopped_rate']}", "",
        "| column | kind | missing % | distinct | min | mean | max / top |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in stats["columns"].items():
        top = c.get("max")
        if c["kind"] == "categorical":
            top = ", ".join(list(c.get("top_values", {}))[:3])
        lines.append(f"| {name} | {c['kind']} | "
                     f"{100 * (c['missing_rate'] or 0):.1f} | {c['distinct']} "
                     f"| {c.get('min', '')} | {c.get('mean', '')} | {top} |")
    path.write_text("\n".join(lines) + "\n")


def save_suite(suite, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite.to_json_dict(), indent=2, default=str))


def freeze(reason=""):
    """Make the current suite and statistics the validation baseline."""
    if not (GENERATED_SUITE.exists() and STATS_JSON.exists()):
        raise FileNotFoundError("run the schema_stats stage first")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GENERATED_SUITE, BASELINE_SUITE)
    shutil.copyfile(STATS_JSON, BASELINE_STATS)
    stats = json.loads(STATS_JSON.read_text())
    info = {"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"),
            "snapshot": stats.get("snapshot"), "source": stats.get("source"),
            "builder": stats.get("builder"),
            "rows": stats["dataset"]["n_rows"], "git_commit": git_commit(),
            "reason": reason}
    BASELINE_INFO.write_text(json.dumps(info, indent=2))
    logger.info("froze baseline from snapshot %s (%d rows) in %s",
                info["snapshot"], info["rows"], BASELINE_DIR)
    return info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", action="store_true",
                        help="copy the current suite and statistics to "
                             "schema/baseline/ (run_stage freeze then "
                             "stores them in GCS)")
    parser.add_argument("--reason", default="",
                        help="why this snapshot is the new baseline")
    args = parser.parse_args()

    with tracked_step("schema_stats") as run:
        if args.freeze:
            run["metrics"].update(freeze(args.reason))
            return
        df = load_features()
        stats = compute_statistics(df, snapshot_info())
        STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
        STATS_JSON.write_text(json.dumps(stats, indent=2))
        write_markdown(stats)
        suite = infer_suite(stats, load_params().get("schema", {}))
        save_suite(suite, GENERATED_SUITE)
        kinds = pd.Series([c["kind"] for c in stats["columns"].values()])
        run["metrics"].update({
            "rows": stats["dataset"]["n_rows"],
            "columns": stats["dataset"]["n_columns"],
            "expectations": len(suite.expectations),
            "stopped_rate": stats["dataset"]["stopped_rate"],
            **{f"columns_{k}": int(v) for k, v in
               kinds.value_counts().items()}})
        run["outputs"]["reports/statistics/statistics.json"] = \
            fingerprint(STATS_JSON)
        run["outputs"]["schema/generated/expectation_suite.json"] = \
            fingerprint(GENERATED_SUITE)
        logger.info("statistics for %d columns; schema with %d expectations",
                    stats["dataset"]["n_columns"], len(suite.expectations))
        if not BASELINE_SUITE.exists():
            logger.warning("no baseline schema yet: validation will use this "
                           "snapshot's schema. Freeze a trusted snapshot with "
                           "`./composer/deploy.sh freeze`")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
