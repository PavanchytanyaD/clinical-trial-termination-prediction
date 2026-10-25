"""Model-level bias report: how well a model does on each group (Fairlearn).

Needs finished trials with a label and a model score (y_score, the predicted
probability that the trial stops). Fairlearn's MetricFrame computes each
metric overall and per group; the gap for a metric is the best group minus
the worst group, using only groups with enough trials.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import average_precision_score, precision_score, \
    recall_score

from src.bias.slice_report import explode_slice


def metric_functions(threshold):
    """Metrics in Fairlearn's form f(y_true, y_score)."""
    def n_trials(y_true, y_score):
        return len(y_true)

    def stopped_rate(y_true, y_score):
        return np.mean(y_true)

    def pr_auc(y_true, y_score):
        # PR-AUC needs both stopped and completed trials in the group.
        if len(set(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_score)

    def recall(y_true, y_score):
        return recall_score(y_true, y_score >= threshold, zero_division=0)

    def precision(y_true, y_score):
        return precision_score(y_true, y_score >= threshold, zero_division=0)

    def flagged_rate(y_true, y_score):
        return np.mean(y_score >= threshold)

    return {"n_trials": n_trials, "stopped_rate": stopped_rate,
            "pr_auc": pr_auc, "recall": recall, "precision": precision,
            "flagged_rate": flagged_rate}


def model_slice_report(trials, slice_names, threshold, min_group_size):
    """Return (per-group metrics table, gaps table) for all slices."""
    metrics = metric_functions(threshold)
    tables, gaps = [], []
    for name in slice_names:
        rows = explode_slice(trials, name)
        frame = MetricFrame(metrics=metrics, y_true=rows.label,
                            y_pred=rows.y_score,
                            sensitive_features=rows[name])
        table = frame.by_group.reset_index().rename(columns={name: "group"})
        table.insert(0, "slice", name)
        tables.append(table)

        big = table[table.n_trials >= min_group_size]
        metric_cols = ["pr_auc", "recall", "precision", "flagged_rate"]
        gap = big[metric_cols].max() - big[metric_cols].min()
        gaps.append({"slice": name, "n_groups_used": len(big), **gap})

    # Overall metrics on the original rows (exploding repeats some trials).
    overall = pd.DataFrame([{
        key: f(trials.label.to_numpy(), trials.y_score.to_numpy())
        for key, f in metrics.items()}])
    overall.insert(0, "slice", "overall")
    overall.insert(1, "group", "ALL")
    return (pd.concat([overall] + tables, ignore_index=True),
            pd.DataFrame(gaps))
