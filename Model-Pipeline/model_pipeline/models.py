"""Candidate model families, their hyperparameter searches and scoring.

To add a model, write a build function that returns an object with
fit(X, y, sample_weight) and predict_proba(X), add it to FAMILIES, and add
its search space to params.yaml under models. An entry there can reuse a
family under another name (family: xgboost) and set text_features: true to
also get the BioBERT text features.
"""

import itertools
import random

import numpy as np


def build_logreg(hp, y, seed):
    """Scaled logistic regression, as in the data pipeline's baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(hp.get("C", 1.0)),
                           class_weight="balanced", max_iter=2000,
                           random_state=seed))


def build_xgboost(hp, y, seed):
    """Gradient-boosted trees; scale_pos_weight balances the classes."""
    from xgboost import XGBClassifier

    pos = max(int(np.sum(y)), 1)
    return XGBClassifier(**hp, scale_pos_weight=(len(y) - pos) / pos,
                         eval_metric="aucpr", tree_method="hist",
                         n_jobs=-1, random_state=seed)


def build_nn(hp, y, seed):
    """Feed-forward network: scaled inputs, ReLU layers, L2 penalty (alpha)
    and early stopping on a held-out 10% of the training set. A single
    network's validation PR-AUC moves by about 0.02 with the random seed,
    so n_seeds networks with different seeds are trained and their
    probabilities averaged (soft voting). MLPClassifier has no
    class_weight, so fit() weights the classes to balance them
    (balance_classes)."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def network(s):
        return MLPClassifier(
            hidden_layer_sizes=tuple(hp.get("hidden_layer_sizes", (128,))),
            alpha=float(hp.get("alpha", 1.0)),
            learning_rate_init=float(hp.get("learning_rate_init", 1e-3)),
            batch_size=int(hp.get("batch_size", 256)),
            max_iter=int(hp.get("max_iter", 200)),
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=s)

    n = int(hp.get("n_seeds", 5))
    model = make_pipeline(
        StandardScaler(),
        VotingClassifier([(f"seed{i}", network(seed + i)) for i in range(n)],
                         voting="soft", n_jobs=-1))
    model.balance_classes = True
    return model


FAMILIES = {"logreg": build_logreg, "xgboost": build_xgboost,
            "nn": build_nn}


def balanced_weights(y, weights=None):
    """Per-trial weights that give both classes the same total weight,
    times the given weights."""
    y = np.asarray(y)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    out = w.copy()
    for cls in np.unique(y):
        mask = y == cls
        out[mask] *= w.sum() / (len(np.unique(y)) * w[mask].sum())
    return out


def fit(model, X, y, weights=None):
    if getattr(model, "balance_classes", False):
        weights = balanced_weights(y, weights)
    if hasattr(model, "steps"):   # sklearn Pipeline: weight the last step
        last = model.steps[-1][0]
        return model.fit(X, y, **{f"{last}__sample_weight": weights})
    return model.fit(X, y, sample_weight=weights)


def param_grid(spec, seed):
    """All combinations (search: grid) or n_iter distinct random ones
    (search: random) of the search space."""
    space = spec.get("space", {})
    keys = list(space)
    combos = [dict(zip(keys, values))
              for values in itertools.product(*(space[k] for k in keys))]
    if spec.get("search", "grid") == "random":
        n = min(int(spec.get("n_iter", 10)), len(combos))
        combos = random.Random(seed).sample(combos, n)
    return combos or [{}]


def predict(model, X):
    return model.predict_proba(X)[:, 1]


def shift_for_groups(scores, groups, cuts, threshold):
    """Apply per-group decision thresholds (the data pipeline's
    group_thresholds mitigation) to raw scores."""
    if not cuts:
        return scores
    from scripts.bias_model import shift_scores
    return shift_scores(scores, np.asarray(groups), cuts, threshold)


def score_candidate(candidate, model, rows, columns, threshold):
    """Scores of one candidate on a split, with its mitigation applied."""
    s = predict(model, rows[columns].to_numpy(dtype=float))
    if candidate.get("cuts"):
        s = shift_for_groups(s, rows[candidate["threshold_slice"]],
                             candidate["cuts"], threshold)
    return s
