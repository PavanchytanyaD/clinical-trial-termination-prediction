"""Bias mitigation for the data-level slices (guideline section 3, part 3).

Starts from the per-trial slice table written by bias detection
(one row per trial: nct_id, label, and one column per slice).

Two methods, chosen in the config:
  reweight  - give each trial a weight so small groups count more
  resample  - draw a weighted resample of the same size

before_after() shows how each group's share and stopped rate change,
so we can prove the mitigation did what we wanted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"
# A share within 0.1% of min_share counts as reaching it (avoids rounding flags).
SHARE_TOL = 1e-3


def labeled_trials(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Keep only finished trials (label 0 or 1). Unfinished trials are censored."""
    if label_col not in df.columns:
        raise KeyError(f"label column '{label_col}' not found in slice table")
    out = df[df[label_col].notna()].copy()
    out[label_col] = out[label_col].astype(int)
    return out.reset_index(drop=True)


def split_values(series: pd.Series, sep: str = "|") -> pd.Series:
    """Turn 'a|b' into ['a', 'b']. Missing or empty values become [UNKNOWN]."""

    def _split(value):
        if pd.isna(value) or str(value).strip() == "":
            return [UNKNOWN]
        parts = [p.strip() for p in str(value).split(sep) if p.strip()]
        return parts or [UNKNOWN]

    return series.map(_split)


def group_shares(df: pd.DataFrame, slice_col: str, sep: str = "|", weights=None) -> pd.Series:
    """Share of (weighted) trials in each group of one slice.

    A trial in several groups (e.g. two therapeutic areas) counts once in
    each, so shares of a multi-value slice can add up to more than 1.
    """
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)
    exploded = pd.DataFrame({"group": split_values(df[slice_col], sep).values, "w": w}).explode(
        "group"
    )
    return exploded.groupby("group")["w"].sum() / w.sum()


def compute_weights(
    df: pd.DataFrame,
    slices: list[str],
    min_share: float,
    max_weight: float,
    sep: str = "|",
    skip_groups: tuple[str, ...] = (UNKNOWN,),
    max_rounds: int = 20,
) -> tuple[pd.Series, list[dict]]:
    """Give trials in small groups a weight above 1.

    For every slice, a group with share s below min_share (m) has its trials'
    weights multiplied by m(1 - s) / (s(1 - m)), which lifts it to m. Boosting
    one slice slightly shifts the others, so this repeats until every group is
    at least m (or max_rounds is reached). A trial in several small groups of
    one slice uses its largest factor. Weights are capped at max_weight, so a
    very tiny group may stay below m, and then scaled to mean 1 so the total
    size of the data stays the same.

    Returns the weights (same index as df) and a list describing each boosted group.
    """
    if not 0 < min_share < 1:
        raise ValueError("min_share must be between 0 and 1")
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1")
    if df.empty:
        raise ValueError("no finished trials to weight")
    for col in slices:
        if col not in df.columns:
            raise KeyError(f"slice column '{col}' not found in slice table")

    tol = SHARE_TOL * min_share
    groups = {col: split_values(df[col], sep) for col in slices}
    w = np.ones(len(df))

    for _ in range(max_rounds):
        changed = False
        for col in slices:
            shares = group_shares(df, col, sep, w)
            small = shares[(shares < min_share - tol) & (~shares.index.isin(skip_groups))]
            if small.empty:
                continue
            group_factor = (min_share * (1 - small) / (small * (1 - min_share))).to_dict()
            factor = groups[col].map(lambda gs: max(group_factor.get(g, 1.0) for g in gs))
            new_w = np.minimum(w * factor.to_numpy(dtype=float), max_weight)
            if not np.allclose(new_w, w):
                changed = True
            w = new_w
        if not changed:
            break

    w = w * len(w) / w.sum()  # mean 1; all weights were >= 1, so max stays <= max_weight
    weights = pd.Series(w, index=df.index, name="weight")

    boosted: list[dict] = []
    for col in slices:
        shares = group_shares(df, col, sep)
        small = shares[(shares < min_share - tol) & (~shares.index.isin(skip_groups))]
        for group, share in small.items():
            in_group = groups[col].map(lambda gs, g=group: g in gs)
            boosted.append(
                {
                    "slice": col,
                    "group": group,
                    "share_before": float(share),
                    "mean_weight": float(weights[in_group].mean()),
                }
            )
    return weights, boosted


def effective_sample_size(weights) -> float:
    """Kish effective sample size. Lower than n when weights are uneven (the cost of reweighting)."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())


def resample(df: pd.DataFrame, weights: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Weighted resample with replacement, same size as df. Same seed gives the same result."""
    return df.sample(n=len(df), replace=True, weights=weights, random_state=seed).reset_index(
        drop=True
    )


def _group_table(df: pd.DataFrame, col: str, label_col: str, sep: str, weights) -> pd.DataFrame:
    """Per-group trial count, weighted share and weighted stopped rate for one slice."""
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)
    ex = pd.DataFrame(
        {
            "group": split_values(df[col], sep).values,
            "w": w,
            "y": df[label_col].to_numpy(dtype=float),
        }
    ).explode("group")
    ex["wy"] = ex["w"] * ex["y"]
    table = ex.groupby("group").agg(n_trials=("w", "size"), w_sum=("w", "sum"), wy_sum=("wy", "sum"))
    table["share"] = table["w_sum"] / w.sum()
    table["stopped_rate"] = table["wy_sum"] / table["w_sum"]
    return table[["n_trials", "share", "stopped_rate"]]


def before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    slices: list[str],
    label_col: str,
    min_share: float,
    sep: str = "|",
    after_weights=None,
) -> pd.DataFrame:
    """Compare every group before mitigation (unweighted) and after it.

    For reweighting, pass the same data as `after` plus `after_weights`.
    For resampling, pass the resampled data as `after` and no weights.
    """
    parts = []
    for col in slices:
        b = _group_table(before, col, label_col, sep, None)
        a = _group_table(after, col, label_col, sep, after_weights)
        t = b.join(a[["share", "stopped_rate"]], how="outer", lsuffix="_before", rsuffix="_after")
        t = t.reset_index()
        t.insert(0, "slice", col)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["n_trials"] = out["n_trials"].fillna(0).astype(int)
    tol = SHARE_TOL * min_share
    out["too_small_before"] = out["share_before"] < min_share - tol
    out["too_small_after"] = out["share_after"] < min_share - tol
    return out


def overall_stopped_rate(df: pd.DataFrame, label_col: str, weights=None) -> float:
    """Weighted share of stopped trials across all finished trials."""
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)
    return float((w * df[label_col].to_numpy(dtype=float)).sum() / w.sum())