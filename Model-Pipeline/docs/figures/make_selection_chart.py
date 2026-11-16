"""Draws selection_ranking.png from a run's reports/selection.json:
    python make_selection_chart.py selection.json selection_ranking.png"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sel = json.load(open(sys.argv[1]))
c = sorted(sel["candidates"], key=lambda t: t["rank"])
names = [t["name"].replace("+thresholds_region_phase", " + thresholds (region, phase)")
         .replace("+thresholds_region", " + thresholds (region)")
         .replace("+reweighted", " + reweighted") for t in c]
win = sel["selected"]["name"]
SURF, INK, INK2, MUTED, GREY, ACC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#c9c8c1", "#2a78d6"
cols = [ACC if t["name"] == win else GREY for t in c]
panels = [("Validation PR-AUC", [t["pr_auc"] for t in c], (0, 0.52), [(0.188, "random 0.188"), (0.30, "gate 0.30")]),
          ("Largest gating-slice gap", [t["max_gap"] for t in c], (0, 0.30), [(0.10, "alert 0.10"), (0.25, "block 0.25")]),
          ("Selection score = PR-AUC − 0.5 × gap", [t["score"] for t in c], (0.33, 0.40), [])]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(15, 7.2), sharey=True,
                         gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.12})
fig.patch.set_facecolor(SURF)
y = list(range(len(c)))[::-1]
for ax, (title, vals, lim, refs) in zip(axes, panels):
    ax.set_facecolor(SURF)
    if lim[0] > 0:   # score: dots, so a non-zero axis does not exaggerate bars
        ax.hlines(y, lim[0], vals, color="#ecebe6", linewidth=1)
        ax.scatter(vals, y, s=[90 if k == ACC else 55 for k in cols], color=cols,
                   edgecolor=SURF, linewidth=2, zorder=3)
    else:
        ax.barh(y, vals, height=0.62, color=cols, edgecolor=SURF, linewidth=2)
    for i, v in zip(y, vals):
        ax.text(v + (lim[1] - lim[0]) * (0.025 if lim[0] > 0 else 0.012), i, f"{v:.3f}", va="center", fontsize=8.5,
                color=INK if cols[y.index(i)] == ACC else INK2,
                fontweight="bold" if cols[y.index(i)] == ACC else "normal")
    for k, (x, lab) in enumerate(refs):
        ax.axvline(x, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(x + (-0.006 if k == 0 else 0.006) * (lim[1] - lim[0]) / 0.3, len(c) - 0.35, lab,
                ha="right" if k == 0 else "left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(*lim)
    ax.set_title(title, fontsize=11, color=INK, pad=22, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe6", linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{t['rank']:>2}. {n}" for t, n in zip(c, names)])
for lab in axes[0].get_yticklabels():
    if lab.get_text().endswith(" xgboost_text"):
        lab.set_fontweight("bold"); lab.set_color(ACC)
axes[2].set_xlabel("ranking (all 16 passed the validation and bias gates)")
fig.suptitle("Model selection: 16 candidates on the validation set, ranked by selection score",
             x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.995)
fig.text(0.01, 0.945, "Selected: xgboost_text (blue). The neural network has a higher PR-AUC (0.480) "
         "but a larger region gap (0.190), so it ranks 2nd. Run run-20261115-200412.",
         fontsize=10, color=INK2)
fig.subplots_adjust(left=0.24, right=0.985, top=0.86, bottom=0.08)
fig.savefig(sys.argv[2], dpi=110, facecolor=SURF)
