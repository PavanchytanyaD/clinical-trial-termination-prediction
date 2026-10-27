# Evidence for the bias report

This folder is not a second report. The report is
[docs/bias_report.md](../bias_report.md). The files here are the raw output
the pipeline wrote, copied from the two runs the report quotes, so every
number in it can be checked.

| Folder | Pipeline stage | Run | Report sections |
|---|---|---|---|
| [data_full_aact/](data_full_aact/) | `bias` (data slicing and reweighting) | full AACT pull, 608,221 trials | 1–6 |
| [model_2019/](model_2019/) | `bias_model` (baseline model per group) | interventional trials registered in 2019, 24,236 trials | 7 |

Every pipeline run writes the same files again, into
`Data-Pipeline/reports/bias/` and `Data-Pipeline/reports/bias_model/`.
`model_2019/bias_model.md` is the table summary that the stage generates
automatically for each run.
