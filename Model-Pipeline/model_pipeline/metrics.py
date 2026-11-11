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
