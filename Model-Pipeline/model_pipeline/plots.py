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


def bias_gaps(gaps, metric, path):
    """Bars of the best-minus-worst group gap per slice and candidate."""
    if gaps.empty:
        return
    pivot = gaps.pivot_table(index="slice", columns="model", values=metric)
    ax = pivot.plot.bar(figsize=(10, 4.5))
    ax.set_ylabel(f"{metric} gap (best group - worst group)")
    ax.set_title(f"{metric} gap per slice (validation set)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")


def bars(series, path, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(series))))
    series.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
