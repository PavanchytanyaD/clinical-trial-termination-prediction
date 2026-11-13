"""Metrics for a stopped-early classifier (label 1 = stopped)."""

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def precision_at_k(y, s, share):
    """Share of truly stopped trials among the top `share` of scores."""
    k = max(1, int(round(len(s) * share)))
    top = np.argsort(-s)[:k]
    return float(np.mean(np.asarray(y)[top]))


def expected_calibration_error(y, p, bins=10):
    """Mean |stop rate - mean predicted probability| over quantile bins of
    the predictions (equal predictions share a bin), weighted by bin size."""
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1))[1:-1])
    groups = np.digitize(p, edges, right=True)
    gaps = [(groups == g).sum() * abs(y[groups == g].mean()
                                      - p[groups == g].mean())
            for g in np.unique(groups)]
    return float(sum(gaps) / len(p))


def compute(y, s, threshold=0.5, top_k_share=0.1):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    both = len(set(y)) == 2
    pred = s >= threshold
    return {
        "pr_auc": float(average_precision_score(y, s)) if both else None,
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision_at_k": precision_at_k(y, s, top_k_share),
        "flagged_rate": float(np.mean(pred)),
        "stopped_rate": float(np.mean(y)),
        "n": int(len(y)),
    }
