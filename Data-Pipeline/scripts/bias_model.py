"""Pipeline stage: baseline model and its bias across subgroups.

The bias stage looks at the data: how many trials each group has and how
often they stop. This stage checks a model: does a simple model predict
equally well for every group?

1. Baseline model: logistic regression on the structured features of the
   finished trials (label 1 = stopped early). The columns preprocess marks
   as a leakage risk are left out. The split is by time: the most recent
   test_share of trials (by registration date) is the test set, as the
   real model will predict for new trials.
2. Per-group evaluation: Fairlearn's MetricFrame
   (src/bias/model_slices.py) computes PR-AUC, recall, precision and the
   share of trials flagged for every group of every slice (phase, sponsor
   class, region, ...). The gap of a metric is the best group minus the
   worst, over groups with at least min_group_size test trials and
   min_stopped stopped ones; UNKNOWN groups (missing values) are reported
   but left out of the gap, as in the data-level report. A recall or PR-AUC
   gap above max_gap is flagged.
3. Mitigation, two ways, each evaluated the same way:
   - reweighted: the model trained with the trial weights from the bias
     stage (under-represented groups weighted up);
   - group_thresholds: the baseline with one decision threshold per group
     of threshold_slice (default region), chosen on the training trials so
     that every group reaches the same recall as the model overall (equal
     opportunity). The report shows the recall gap it closes and what it
     costs in precision.

Outputs in reports/bias_model/: summary.json (overall metrics and gaps of
both models), model_slices.csv (every group, both models), model_gaps.csv
(gap per slice, both models) and bias_model.md (readable report).
Settings: params.yaml (section bias_model) and configs/bias_config.yaml
(slices, threshold, min_group_size).

    python -m scripts.bias_model
"""

import json
import logging

import numpy as np
import pandas as pd

from scripts.bias import bias_config
from scripts.config import (BIAS_DIR, BIAS_MODEL_DIR, FEATURES_CSV,
                            FEATURES_META, TRIAL_SLICES, load_params)
from scripts.tracking import fingerprint, tracked_step

logger = logging.getLogger("bias_model")

DEFAULTS = {"test_share": 0.2, "max_gap": 0.10, "min_stopped": 10,
            "threshold_slice": "region", "regularization": 1.0,
            "seed": 42}
METRICS = ["pr_auc", "recall", "precision", "flagged_rate"]
WEIGHTS = BIAS_DIR / "mitigation" / "trial_weights.csv"


def feature_columns(df, meta):
    """Numeric features, without ids, dates, the label and leakage risks."""
    skip = {"nct_id", "overall_status", "label", "start_date",
            "study_first_submitted_date", *meta.get("leakage_risk", [])}
    listed = meta.get("feature_columns") or list(df.columns)
    return [c for c in listed if c in df.columns and c not in skip
            and pd.api.types.is_numeric_dtype(df[c])]


def time_split(df, test_share):
    """(train, test): the most recent trials by registration date are the
    test set."""
    order = pd.to_datetime(df["study_first_submitted_date"],
                           errors="coerce").fillna(pd.Timestamp.min)
    df = df.assign(_order=order).sort_values(["_order", "nct_id"])
    n_test = max(1, int(round(len(df) * test_share)))
    return (df.iloc[:-n_test].drop(columns="_order"),
            df.iloc[-n_test:].drop(columns="_order"))


def train(train_df, columns, weights=None, regularization=1.0, seed=42):
    """Scaled logistic regression; class_weight balances stopped and
    completed trials, weights (per trial) come from the mitigation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=regularization, class_weight="balanced",
                           max_iter=2000, random_state=seed))
    model.fit(train_df[columns].to_numpy(dtype=float),
              train_df["label"].astype(int),
              logisticregression__sample_weight=weights)
    return model


def group_thresholds(scores, labels, groups, threshold, min_stopped):
    """One threshold per group, so each group's recall on these trials
    matches the overall recall at the global threshold. Groups with fewer
    than min_stopped stopped trials, and UNKNOWN, keep the global one."""
    target = float(np.mean(scores[labels == 1] >= threshold))
    cuts = {}
    for group in pd.unique(groups):
        stopped = scores[(groups == group) & (labels == 1)]
        if group != "UNKNOWN" and len(stopped) >= min_stopped:
            cuts[group] = float(np.quantile(stopped, 1 - target))
    return cuts


def shift_scores(scores, groups, cuts, threshold):
    """Scores moved so that score >= threshold exactly when the trial is at
    or above its group's cut (the order within a group is kept)."""
    eps = 1e-6
    p = np.clip(scores, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    cut = np.array([cuts.get(g, threshold) for g in groups])
    cut = np.clip(cut, eps, 1 - eps)
    base = np.log(threshold / (1 - threshold))
    shifted = logit - np.log(cut / (1 - cut)) + base
    return 1 / (1 + np.exp(-shifted))


def gaps_by_slice(table, min_group_size, min_stopped):
    """Best minus worst group per metric and slice, over groups that are
    big enough and not UNKNOWN."""
    rows = []
    stopped = table["n_trials"] * table["stopped_rate"]
    use = table[(table["slice"] != "overall") & (table["group"] != "UNKNOWN")
                & (table["n_trials"] >= min_group_size)
                & (stopped.round() >= min_stopped)]
    for (model, name), g in use.groupby(["model", "slice"], sort=False):
        rows.append({"model": model, "slice": name,
                     "n_groups_used": len(g),
                     **{m: g[m].max() - g[m].min() for m in METRICS}})
    return pd.DataFrame(rows, columns=["model", "slice", "n_groups_used",
                                       *METRICS])


def overall_metrics(y_true, y_score, threshold):
    from sklearn.metrics import (average_precision_score, precision_score,
                                 recall_score, roc_auc_score)
    both = len(set(y_true)) == 2
    pred = y_score >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, y_score))
        if both else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both else None,
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "flagged_rate": float(np.mean(pred)),
        "stopped_rate": float(np.mean(y_true)),
    }


def with_slices(df, slices, slice_names):
    out = df[["nct_id", "label"]].merge(slices, on="nct_id", how="left")
    for name in slice_names:
        out[name] = out[name].fillna("UNKNOWN").astype(str)
    out["label"] = out["label"].astype(int)
    return out


def evaluate(scores, test, threshold, slice_names, min_group_size,
             min_stopped):
    """Per-group metrics and gaps of each model's test scores."""
    from src.bias.model_slices import model_slice_report

    tables, overall = [], {}
    for name, y_score in scores.items():
        rows = test.assign(y_score=y_score)
        table, _ = model_slice_report(rows, slice_names, threshold,
                                      min_group_size)
        tables.append(table.assign(model=name))
        overall[name] = overall_metrics(rows["label"].to_numpy(), y_score,
                                        threshold)
    table = pd.concat(tables, ignore_index=True)
    return table, gaps_by_slice(table, min_group_size, min_stopped), overall


def markdown(summary, gaps):
    lines = ["# Baseline model: performance per subgroup", "",
             f"Logistic regression on {summary['n_features']} structured "
             f"features; trained on {summary['n_train']} finished trials, "
             f"tested on the {summary['n_test']} most recently registered "
             f"(threshold {summary['threshold']}). Models: baseline; "
             f"reweighted (trained with the bias stage's trial weights); "
             f"group_thresholds (baseline with one threshold per "
             f"{summary['threshold_slice']} group, for equal recall).", "",
             "## Overall (test set)", "",
             "| Model | PR-AUC | ROC-AUC | Recall | Precision | Flagged |",
             "|---|---|---|---|---|---|"]
    for name, m in summary["overall"].items():
        cells = [m.get(k) for k in ("pr_auc", "roc_auc", "recall",
                                    "precision", "flagged_rate")]
        lines.append(f"| {name} | " + " | ".join(
            "n/a" if v is None else f"{v:.3f}" for v in cells) + " |")
    lines += ["", f"Stopped rate in the test set: "
              f"{summary['overall']['baseline']['stopped_rate']:.3f}.", "",
              "## Gap between the best and the worst group", "",
              f"Groups with at least {summary['min_group_size']} test "
              f"trials and {summary['min_stopped']} stopped ones, without "
              f"UNKNOWN; a recall or PR-AUC gap above {summary['max_gap']} "
              f"is flagged.", "",
              "| Slice | Model | Groups | " + " | ".join(METRICS) + " |",
              "|---|---|---|" + "---|" * len(METRICS)]
    for row in gaps.itertuples():
        values = [getattr(row, m) for m in METRICS]
        lines.append(f"| {row.slice} | {row.model} | {row.n_groups_used} | "
                     + " | ".join("n/a" if pd.isna(v) else f"{v:.3f}"
                                  for v in values) + " |")
    flagged = summary["flagged"]
    lines += ["", "## Flagged", ""]
    lines += ([f"- {f['model']}: {f['slice']} {f['metric']} gap "
               f"{f['gap']:.3f}" for f in flagged] or ["None."])
    return "\n".join(lines) + "\n"


def run(params, config):
    p = {**DEFAULTS, **params}
    model_cfg = config.get("model", {})
    threshold = float(model_cfg.get("threshold", 0.5))
    min_group_size = int(model_cfg.get("min_group_size", 100))
    slice_names = [s for s in config["slices"]["names"]
                   if s != "study_type"]  # one value among finished trials

    features = pd.read_csv(FEATURES_CSV, dtype={"nct_id": str})
    meta = json.loads(FEATURES_META.read_text())
    finished = features[features["label"].notna()].copy()
    if finished["label"].nunique() < 2:
        raise ValueError("the finished trials have only one class; "
                         "no model can be trained")
    columns = feature_columns(finished, meta)
    train_df, test = time_split(finished, p["test_share"])
    if train_df["label"].nunique() < 2:
        raise ValueError("the training trials have only one class")

    weights = pd.read_csv(WEIGHTS, dtype={"nct_id": str}) \
        if WEIGHTS.exists() else pd.DataFrame(columns=["nct_id", "weight"])
    w = train_df[["nct_id"]].merge(weights, on="nct_id", how="left")
    w = w["weight"].fillna(1.0).to_numpy(dtype=float)

    slices = pd.read_csv(TRIAL_SLICES, dtype={"nct_id": str})
    slice_names = [s for s in slice_names if s in slices.columns]
    slices = slices[["nct_id"] + slice_names]
    train_rows = with_slices(train_df, slices, slice_names)
    test_rows = with_slices(test, slices, slice_names)

    def x(df):
        return df[columns].to_numpy(dtype=float)

    baseline = train(train_df, columns, None, p["regularization"], p["seed"])
    reweighted = train(train_df, columns, w, p["regularization"], p["seed"])
    base_test = baseline.predict_proba(x(test))[:, 1]
    scores = {"baseline": base_test,
              "reweighted": reweighted.predict_proba(x(test))[:, 1]}
    cut_slice = p["threshold_slice"]
    cuts = {}
    if cut_slice in slice_names:
        cuts = group_thresholds(baseline.predict_proba(x(train_df))[:, 1],
                                train_rows["label"].to_numpy(),
                                train_rows[cut_slice].to_numpy(), threshold,
                                p["min_stopped"])
        scores["group_thresholds"] = shift_scores(
            base_test, test_rows[cut_slice].to_numpy(), cuts, threshold)
    table, gaps, overall = evaluate(scores, test_rows, threshold,
                                    slice_names, min_group_size,
                                    p["min_stopped"])

    flagged = [{"model": row.model, "slice": row.slice, "metric": m,
                "gap": float(getattr(row, m))}
               for row in gaps.itertuples() for m in ("pr_auc", "recall")
               if pd.notna(getattr(row, m)) and getattr(row, m) > p["max_gap"]]
    summary = {
        "n_train": len(train_df), "n_test": len(test),
        "n_features": len(columns), "threshold": threshold,
        "min_group_size": min_group_size, "min_stopped": p["min_stopped"],
        "max_gap": p["max_gap"], "threshold_slice": cut_slice,
        "group_thresholds": cuts,
        "upweighted_trials": int((w > 1.01).sum()),
        "overall": overall, "flagged": flagged,
    }
    BIAS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BIAS_MODEL_DIR / "model_slices.csv", index=False)
    gaps.to_csv(BIAS_MODEL_DIR / "model_gaps.csv", index=False)
    (BIAS_MODEL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2))
    (BIAS_MODEL_DIR / "bias_model.md").write_text(markdown(summary, gaps))
    for f in flagged:
        logger.warning("%s model: %s %s gap %.3f above %.2f", f["model"],
                       f["slice"], f["metric"], f["gap"], p["max_gap"])
    return summary


def main():
    with tracked_step("bias_model") as run_record:
        summary = run(load_params().get("bias_model", {}), bias_config())
        for name, m in summary["overall"].items():
            run_record["metrics"].update(
                {f"{name}_{k}": v for k, v in m.items() if v is not None})
        run_record["metrics"]["flagged_gaps"] = len(summary["flagged"])
        run_record["outputs"]["reports/bias_model"] = \
            fingerprint(BIAS_MODEL_DIR)
        logger.info("baseline model: PR-AUC %s on %d test trials, %d gaps "
                    "flagged", summary["overall"]["baseline"]["pr_auc"],
                    summary["n_test"], len(summary["flagged"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
