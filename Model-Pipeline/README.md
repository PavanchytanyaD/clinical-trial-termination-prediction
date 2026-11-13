# Model pipeline

Trains, validates, bias-checks, selects and registers the model that
predicts whether a clinical trial will stop early. It reads the exact data
version the [data pipeline](../Data-Pipeline/README.md) published and runs
on Vertex AI as ten stages:

```
load_data → train → validate → bias → select → calibrate → test → register
                                                              ├→ lead_time
                                                              └→ sensitivity
```

| Stage | What it does | Outputs (in the run's work folder) |
|---|---|---|
| `load_data` | Reads a snapshot's `dvc.lock` from the bucket and downloads the features, slices, trial weights, BioBERT embeddings and AACT studies table by md5 from the DVC remote. Splits by registration date: oldest 60% train, next 20% validation, newest 20% test. Reduces each text field's 768-number embedding to 16 PCA components fitted on the training split (`txt_<field>_<i>`). Keeps stop dates and final status beside the features for `lead_time` (never as features). | `data/`, `data/data_version.json`, `data/text_pca.joblib` |
| `train` | Hyperparameter search per entry in `params.yaml` → `models` (LR, XGBoost, XGBoost with text, neural network), every trial logged to MLflow; best per entry kept, plus a version trained with the bias stage's trial weights. | `models/`, `reports/tuning_results.csv` |
| `validate` | Every candidate on the validation set: PR-AUC, ROC-AUC, Brier, precision, recall, F1, Precision@K. | `reports/val_metrics.json`, comparison charts |
| `bias` | Fairlearn per-group metrics on 7 slices (reusing `src/bias/model_slices.py`); adds a per-region threshold candidate for each model. | `reports/bias/` (report, tables, charts) |
| `select` | Eligible if PR-AUC and ROC-AUC pass the gate and the largest subgroup gap is under the block level; winner = highest PR-AUC − 0.5 × gap. Stops the run if none passes. | `reports/selection.md` |
| `calibrate` | Fits sigmoid and isotonic calibrators to the selected model's validation scores, keeps the one with the lower 5-fold cross-validated Brier score. Decisions still use the raw score; the calibrated probability is the risk shown to reviewers. | `models/calibrator.joblib`, `reports/calibration/` |
| `test` | Selected model on the untouched test set; must pass the gate again; reports Brier and calibration error before and after calibration; packages the model with its calibrator (and the text PCA if it uses text). | `reports/test/`, `artifact/` |
| `lead_time` | For stopped trials the model flags on the test set: months from registration to stop (median, IQR, share over 12 months). Error analysis: recall per final status and per time to stop, top false positives and negatives. | `reports/lead_time/` |
| `sensitivity` | SHAP feature importance (and per text field for a model with text features); hyperparameter sensitivity from the search. | `reports/sensitivity/` |
| `register` | New version in the Vertex AI Model Registry (alias `challenger`); `champion` moves only if the test PR-AUC is not worse than the current champion's. | `reports/registry.json` |

Email alerts (same subscribers as the data pipeline) on: a failed stage, no
model passing the gate, bias above the alert level, a model failing on the
test set, promotion, and a model not promoted.

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
| Neural network with BioBERT text features | `models.py` (`build_nn`), `data.py` (`text_features`) |
| Calibration | `calibrate.py`, `reports/calibration/`, `reports/test/` |
| Lead time and error analysis | `lead_time.py`, `reports/lead_time/` |
| Model development writeup | [`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md) |
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
then tuned, validated, bias-checked and compared like the others. An entry
can reuse a family under another name (`family: xgboost`) and add the
BioBERT text features (`text_features: true`).
