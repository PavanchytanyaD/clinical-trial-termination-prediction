# Model pipeline

Trains, validates, bias-checks, selects and registers the model that
predicts whether a clinical trial will stop early. It reads the exact data
version the [data pipeline](../Data-Pipeline/README.md) published and runs
on Vertex AI as eight stages:

```
load_data → train → validate → bias → select → test → register
                                                  └→ sensitivity
```

| Stage | What it does | Outputs (in the run's work folder) |
|---|---|---|
| `load_data` | Reads a snapshot's `dvc.lock` from the bucket and downloads the features, slices and trial weights by md5 from the DVC remote. Splits by registration date: oldest 60% train, next 20% validation, newest 20% test. | `data/`, `data/data_version.json` |
| `train` | Hyperparameter search per model family (`params.yaml` → `models`), every trial logged to MLflow; best per family kept, plus a version trained with the bias stage's trial weights. | `models/`, `reports/tuning_results.csv` |
| `validate` | Every candidate on the validation set: PR-AUC, ROC-AUC, Brier, precision, recall, F1, Precision@K. | `reports/val_metrics.json`, comparison charts |
| `bias` | Fairlearn per-group metrics on 7 slices (reusing `src/bias/model_slices.py`); gap = overall minus worst qualified group, with a bootstrap interval. Adds per-group threshold candidates (region; region then phase), fitted on the validation set with cross-fitting. | `reports/bias/` (report, tables, charts) |
| `select` | Eligible if PR-AUC and ROC-AUC pass the gate and the bias gate does not block it; ranked by PR-AUC − 0.5 × largest gating-slice gap. Stops the run if none passes. | `reports/selection.md` |
| `test` | Selected model on the untouched test set; must pass the performance and bias gates again, otherwise the next ranked candidate is tried (up to 3); packages the model that passes. | `reports/test/`, `artifact/` |
| `sensitivity` | SHAP feature importance; hyperparameter sensitivity from the search. | `reports/sensitivity/` |
| `register` | New version in the Vertex AI Model Registry (alias `challenger`); `champion` moves only if the test PR-AUC is not worse than the current champion's. | `reports/registry.json` |

Email alerts (same subscribers as the data pipeline) on: a failed stage,
candidates blocked by the bias gate or without enough data to check, no
model passing the gate, bias above the alert level, a model failing on the
test set (and the fallback to the next one), promotion, and a model not
promoted.

## Bias gate

Settings in `params.yaml` (`bias` and `gate`); code in
`model_pipeline/gaps.py`.

- **Gap** of a metric (recall, PR-AUC) for a group = overall value minus
  the group's value. Only groups with at least `min_group_size` trials and
  `min_stopped` stopped ones count (UNKNOWN never does), and a slice is
  checked only with at least `min_groups` such groups. Worst group against
  overall, because best minus worst grows with the number of groups and
  often hinges on one small group.
- **Interval**: 90% percentile bootstrap (200 seeded resamples of the
  trials). A gap counts as above a level only if its lower bound is.
- **Gating slices** (`region`, `phase`, `sponsor_class`) block a model above
  `block_bias_gap` (0.25); every slice, including the report slices
  (`therapeutic_area`, `gender`, `age_group`, `start_era`), raises an alert
  above `alert_bias_gap` (0.10).
- **Insufficient data**: if no gating slice can be checked, the candidate
  is blocked (`insufficient_bias_data: block`) instead of passing with a
  gap of 0. Small samples need smaller `min_group_size` / `min_stopped`.
- The same rules apply again on the test set.

## Assignment items

| Guideline item | Where |
|---|---|
| Load data from the data pipeline (2.1, 8.2) | `model_pipeline/data.py` |
| Train and select the best model (2.2, 8.3) | `train.py`, `select.py` |
| Validation on a hold-out set (2.3, 8.4) | `validate.py`, `test_set.py` |
| Bias detection with slicing, Fairlearn (2.4, 2.5, 6, 8.5) | `bias.py`, `reports/bias/bias_report.md` |
| Bias mitigation and trade-offs (6.3, 6.4) | reweighted and group_thresholds candidates, `bias_report.md`, `selection.md` |
| Selection after bias checking (8.6) | `select.py` |
| Push to the Model Registry (2.6, 8.7) | `registry.py` |
| Hyperparameter tuning (3) | `params.yaml` → `models`, `train.py` |
| Experiment tracking, MLflow (4) | `tracking.py` |
| Comparison charts, confusion matrix (4) | `reports/figures/`, `reports/test/` |
| Sensitivity analysis, SHAP and hyperparameters (5) | `sensitivity.py` |
| CI/CD: train, validate, bias, registry on push (7.1 to 7.4) | `.github/workflows/model-pipeline.yml` |
| Alerts (7.5) | `notify.py` |
| Rollback (7.6) | `registry.py` (champion alias, `rollback` command) |
| Docker (8.1) | `Dockerfile`, `cloudbuild.yaml` |

## Run it

**On your machine or Cloud Shell** (no registry):

```bash
cd Data-Pipeline && dvc pull preprocess bias && cd ../Model-Pipeline
pip install pandas scikit-learn fairlearn -r requirements.txt
python -m model_pipeline.run --stage all --source local --skip-register
```

Reports are in `Model-Pipeline/runs/local/`.

**On Vertex AI:**

```bash
gcloud builds submit --config Model-Pipeline/cloudbuild.yaml \
  --substitutions _TAG=$(git rev-parse --short HEAD) .
python Model-Pipeline/vertex/submit.py --snapshot latest
```

Follow the run in the console (Agent Platform → Pipelines). Each run's
work folder is `gs://ctt-mlops-project-data/model-pipeline/runs/<run_id>/`.

**Registry:**

```bash
cd Model-Pipeline
python -m model_pipeline.registry status
python -m model_pipeline.registry rollback --version 2
```

**Tests:** `cd Model-Pipeline && pytest -q tests`

## Settings

Everything is in `params.yaml`: data version, split shares, search spaces,
threshold, slices, gate levels and registry names.

## Adding a model

Write a build function in `model_pipeline/models.py` that returns an object
with `fit(X, y, sample_weight)` and `predict_proba(X)`, add it to
`FAMILIES`, and add its search space under `models` in `params.yaml`. It is
then tuned, validated, bias-checked and compared like the others.
