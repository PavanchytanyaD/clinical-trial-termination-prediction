# Experiment Tracking and Run Comparison (guideline section 4)

Owner: Atharva.

## What is logged

Every pipeline run is one MLflow run in the experiment `ctt-model-development`
(`params.yaml` → `tracking.experiment`). Each hyperparameter trial is a nested
run under it.

| Logged | Where in the code |
|---|---|
| Hyperparameters and validation metrics of every trial (nested runs) | `train.py` → `tracking.log_trial` |
| Validation metrics of each candidate model | `validate.py` |
| Largest bias gap across slices | `bias.py` (`max_bias_gap`) |
| Test-set metrics of the selected model (`test_*`) | `test_set.py` |
| Selected model and registry version (tags) | `select.py`, `registry.py` |
| Plots and reports (artifacts), e.g. sensitivity charts | `sensitivity.py` and others |

## Where runs are logged

`model_pipeline/tracking.py` → `tracking_uri()` picks the first of:

1. `MLFLOW_TRACKING_URI` in the environment (CI, Vertex AI, a shared server)
2. `tracking.uri` in `params.yaml` (the team's shared server or folder)
3. `mlruns/` in the run's own work folder (copied to the bucket)

Before this change only 1 and 3 existed, and 1 was never set, so every run
wrote to its own folder and runs could not be compared. Setting option 1 or 2
puts every run in one place.

## Option A: one shared MLflow server

On the machine that will host it (needs the full `mlflow` package, not the
`mlflow-skinny` the pipeline installs):

```
pip install mlflow google-cloud-storage
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination gs://<bucket>/mlflow-artifacts \
  --host 0.0.0.0 --port 5000
```

Then point the pipeline at it, either per shell:

```
export MLFLOW_TRACKING_URI=http://<host>:5000
```

or for everyone in `params.yaml`:

```yaml
tracking:
  experiment: ctt-model-development
  uri: http://<host>:5000
```

Open `http://<host>:5000`, pick the experiment, select runs and click
**Compare** to see metrics and parameters side by side.

CI and Vertex AI jobs can only log to the server if they can reach it over
the network; otherwise they fall back to their work folder (option 3) and
are compared with option B.

## Option B: compare runs saved in separate folders

Download the `mlruns/` folders of the runs to compare from the bucket, then:

```
cd Model-Pipeline
python -m model_pipeline.compare_runs \
  --uri runs/<run-a>/mlruns --uri runs/<run-b>/mlruns \
  --out reports/run_comparison
```

The same command works with a server: `--uri http://<host>:5000`.

Outputs in `reports/run_comparison/`:

- `run_comparison.csv`: one row per pipeline run (start time, status,
  selected model, registry version, every metric and parameter)
- `run_comparison.png`: `test_pr_auc` per run (`--metric` to change it)

Hyperparameter trials are left out of this table; they are compared within a
run by the tuning and sensitivity charts.

## Tests

`tests/test_tracking_compare.py`: the order of the three tracking addresses,
the file-store flag, passwords kept out of log lines, combining two folders,
skipping nested trials, the output files, and failure on a missing folder.
