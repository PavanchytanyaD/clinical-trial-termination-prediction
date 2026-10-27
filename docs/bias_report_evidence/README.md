# Evidence for the bias report

This folder is not a second report. The report is
[docs/bias_report.md](../bias_report.md). These files are the raw output
of the one pipeline run the report is based on, copied unchanged, so every
number in it can be checked.

- **Run:** `dvc repro bias_model` (stages `acquire` → `preprocess` → `bias` → `bias_model`),
  run ID `realdata_2019`, on 2026-10-26
- **Data:** the real AACT database (snapshot 20261027), interventional trials
  registered in 2019: 24,236 trials, 16,753 finished

| Folder | Written by stage | Same folder in every run | Report sections |
|---|---|---|---|
| [bias/](bias/) | `bias` (data slicing and reweighting) | `Data-Pipeline/reports/bias/` | 1–6 |
| [bias_model/](bias_model/) | `bias_model` (baseline model per group) | `Data-Pipeline/reports/bias_model/` | 7 |

`bias_model/bias_model.md` is the table summary the stage writes by itself.
The per-trial weights (`bias/mitigation/trial_weights.csv`, 16,753 rows) are
left out to keep the repository small; every run writes them.
