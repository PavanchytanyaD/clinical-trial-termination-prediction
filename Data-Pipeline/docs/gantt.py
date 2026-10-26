"""Gantt chart of one Airflow DAG run, from its recorded task times.

    airflow tasks states-for-dag-run ctt_data_pipeline <run_id> -o json > run.json
    python docs/gantt.py run.json "Parallel branches" docs/images/gantt.png

Needs matplotlib (not part of the pipeline image).
"""
import datetime as dt
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["acquire", "preprocess", "bias", "text", "schema_stats", "anomalies",
         "embeddings", "validate", "publish"]
COLORS = {"embeddings": "#534AB7"}

rows = {r["task_id"]: r for r in json.load(open(sys.argv[1]))}
t = {k: (dt.datetime.fromisoformat(r["start_date"]),
         dt.datetime.fromisoformat(r["end_date"])) for k, r in rows.items()}
t0 = min(s for s, _ in t.values())
total = (max(e for _, e in t.values()) - t0).total_seconds()

tasks = [k for k in ORDER if k in t]
fig, ax = plt.subplots(figsize=(10, 4.2))
for i, k in enumerate(tasks):
    s, e = ((x - t0).total_seconds() for x in t[k])
    ax.barh(i, e - s, left=s, height=0.55, color=COLORS.get(k, "#1D9E75"))
    ax.text(e + total * 0.006, i, f"{e - s:.0f} s", va="center", fontsize=9)
ax.set_yticks(range(len(tasks)), tasks)
ax.invert_yaxis()
ax.set_xlabel("seconds since the run started")
ax.set_xlim(0, total * 1.1)
ax.set_title(f"{sys.argv[2]}: whole run {total:.0f} s", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=150)
print(json.dumps({"total": round(total, 1), **{
    k: round((t[k][1] - t[k][0]).total_seconds(), 1) for k in tasks}}))
