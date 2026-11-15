"""Charts for the reports (saved as PNG, also logged to MLflow)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def compare_models(table, path, cols=("pr_auc", "roc_auc", "recall",
                                      "precision", "precision_at_k")):
    """Grouped bars: one group per metric, one bar per candidate."""
    names = list(table)
    x = np.arange(len(cols))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, name in enumerate(names):
        vals = [table[name].get(c) or 0 for c in cols]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2, cols)
    ax.set_ylim(0, 1)
    ax.set_title("Candidates on the validation set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pr_curves(curves, path, title):
    """curves: {name: (y_true, scores)}."""
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def reliability(curves, path, title, bins=10):
    """Predicted probability against the observed stop rate, per bin of
    equal size; curves: {name: (y_true, probabilities)}."""
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    for name, (y, p) in curves.items():
        rate, mean = calibration_curve(y, np.clip(p, 0, 1), n_bins=bins,
                                       strategy="quantile")
        ax.plot(mean, rate, "o-", label=name)
    ax.set_xlabel("Predicted probability of stopping")
    ax.set_ylabel("Share of trials that stopped")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def histogram(values, path, title, xlabel, bins=30):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trials")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confusion(y, s, threshold, path, title):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, (np.asarray(s) >= threshold).astype(int), ax=ax,
        display_labels=["completed", "stopped"], colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bias_gap_bars(slices, metric, names, path, title, alert, block):
    """Worst-group gap per slice (overall minus worst group) for the given
    candidates, with the bootstrap interval as error bars and the alert and
    block levels as lines. Report-only slices are marked "(report)"."""
    rows = slices[(slices["metric"] == metric)
                  & slices["model"].isin(names)]
    if rows.empty:
        return
    order = list(dict.fromkeys(rows.sort_values("role", ascending=True)
                               ["slice"]))
    labels = []
    for s in order:
        r = rows[rows["slice"] == s]
        labels.append(s + ("" if r["role"].iloc[0] == "gate"
                           else "\n(report)")
                      + ("\n(too few groups)" if r["gap"].isna().all()
                         else ""))
    x = np.arange(len(order))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(order)), 4.5))
    for i, name in enumerate(names):
        g = rows[rows["model"] == name].set_index("slice").reindex(order)
        gap = g["gap"].to_numpy(dtype=float)
        err = np.vstack([gap - g["ci_low"].to_numpy(dtype=float),
                         g["ci_high"].to_numpy(dtype=float) - gap])
        ax.bar(x + i * width, np.nan_to_num(gap), width, label=name)
        ax.errorbar(x + i * width, np.nan_to_num(gap),
                    yerr=np.nan_to_num(np.clip(err, 0, None)), fmt="none",
                    ecolor="black", elinewidth=1, capsize=3)
    ax.axhline(alert, color="orange", linestyle="--", linewidth=1,
               label=f"alert {alert}")
    ax.axhline(block, color="red", linestyle="--", linewidth=1,
               label=f"block {block}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(names) - 1) / 2, labels, fontsize=8)
    ax.set_ylabel(f"{metric}: overall - worst group")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bars(series, path, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(series))))
    series.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
