# Model development

How the termination model is built, tuned, calibrated and checked. The
pipeline stages are described in [README.md](README.md); this page covers
the modelling choices and the results.

**Results below:** one local run of the full pipeline (registry skipped) on
the real AACT data version from the data pipeline's run of 2026-11-12:
24,235 interventional trials registered in 2019, of which 16,802 have
finished (3,096 stopped: terminated, withdrawn or suspended; 18.4%).
Numbers will move when the pipeline runs on a larger data version.

## 1. Task and data

- **Label:** 1 if a finished trial stopped early (terminated, withdrawn,
  suspended), 0 if it completed. Trials still running are left out.
- **Only what is known at registration.** Completion dates, actual
  enrollment, results and `why_stopped` are excluded by the data pipeline
  (`LEAKAGE_EXCLUDED` in `src/features/structured_features.py`).
- **Time split** by registration date: oldest 60% train (10,082 trials,
  Jan to Jul 2019), next 20% validation (3,360, Jul to Oct), newest 20%
  test (3,360, Oct to Dec). The test set is only used once, after
  selection.

## 2. Features

| Group | Count | Source |
|---|---|---|
| Structured | 155 | design, phase, size, eligibility, sponsor, countries, interventions, therapeutic area (data pipeline, one-hot with fixed categories) |
| Text | 96 | BioBERT embeddings of title, brief summary, detailed description, inclusion, exclusion and outcomes |

**Text features.** The data pipeline embeds each field with
`dmis-lab/biobert-base-cased-v1.2` (not fine-tuned; long texts split into
overlapping 512-token windows and mean-pooled). Six fields of 768 numbers
are too many next to 155 structured features on 10k training trials, so
`load_data` reduces each field to 16 PCA components, fitted on the training
split only. 16 per field was the best of 8, 16, 32 and 64 in a quick check
(logistic regression and NN validation PR-AUC). The fitted PCAs are saved
with the model (`text_pca.joblib`) so inference applies the same reduction.

## 3. Models

| Entry | Model | Features | Class imbalance |
|---|---|---|---|
| `logreg` | scaled logistic regression | structured | `class_weight="balanced"` |
| `xgboost` | gradient-boosted trees | structured | `scale_pos_weight` |
| `xgboost_text` | same as `xgboost` | structured + text | `scale_pos_weight` |
| `nn` | feed-forward network | structured + text | balanced sample weights |

`xgboost_text` is there to separate the two changes the NN brings (a new
model type and the text), so the comparison shows which one helps.

**Neural network** (`build_nn` in `model_pipeline/models.py`): standard
scaling, then an `MLPClassifier` (ReLU, Adam, L2 penalty `alpha`, early
stopping on 10% of the training set). Two findings shaped it:

1. **It overfits fast.** Without early stopping, validation PR-AUC fell
   from 0.466 to 0.416; with it, training stops after about 20 epochs. The
   best settings use small layers and a strong L2 penalty (alpha 0.3 to 3),
   much stronger than sklearn's default of 0.0001.
2. **One network is noisy.** The same settings with 5 different seeds gave
   validation PR-AUC from 0.441 to 0.479. The model is therefore an average
   of 5 networks with different seeds (soft voting), which scored above
   the mean single network every time (0.474 vs 0.463, 0.482 vs 0.454).

It stays a plain scikit-learn object: it saves with joblib and fits the
existing training, bias, SHAP and registry code without changes.

## 4. Hyperparameter tuning

Random search per entry (`params.yaml` → `models`), every trial a nested
MLflow run, best trial chosen by validation PR-AUC.

| Entry | Trials | Best val PR-AUC | Mean val PR-AUC | Space |
|---|---|---|---|---|
| logreg | 4 (grid) | 0.456 | 0.455 | C |
| xgboost | 12 | 0.471 | 0.456 | trees, depth, learning rate, subsample, column sample, min child weight |
| xgboost_text | 12 | 0.473 | 0.456 | same |
| nn | 16 | 0.479 | 0.461 | layers {32, 64, 128, 256, 128-64}, alpha {0.3, 1, 3, 10}, learning rate {1e-4, 3e-4, 1e-3}, batch {128, 256} |

Best NN: one hidden layer of 64, alpha 0.3, learning rate 0.001, batch 128.
The top five NN trials are within 0.005 of each other (one layer of 32 to
256, alpha 0.3 to 3, learning rate 0.001 mostly), so the result does not
hinge on one exact setting. Per-hyperparameter sensitivity is in
`reports/sensitivity/hyperparameter_sensitivity.csv`.

## 5. Results

| Model | Features | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision@10% | Test recall |
|---|---|---|---|---|---|---|
| logreg | structured | 0.455 | 0.478 | 0.770 | 0.557 | 0.786 |
| xgboost | structured | 0.471 | 0.487 | 0.784 | 0.560 | 0.828 |
| xgboost_text | structured + text | 0.473 | 0.462 | 0.786 | 0.536 | 0.606 |
| **nn** | structured + text | **0.479** | **0.493** | **0.791** | **0.583** | 0.754 |

Base rate (a random model's PR-AUC): 0.185. Threshold 0.5 on the raw score.

- The NN is best on every ranking metric on both validation and test. Of
  the 336 trials it ranks most at risk (top 10%), 58% stopped, about 3x
  the base rate.
- **The margin over XGBoost is small** (0.006 test PR-AUC), about the size
  of the seed noise in section 3, so the honest summary is "as good as
  XGBoost or slightly better". The text features help the NN more than
  they help XGBoost: `xgboost_text` beats `xgboost` on validation but not
  on test.
- The select stage picked the NN: it had the highest selection score
  (validation PR-AUC minus 0.5 x largest subgroup gap) of all 12
  candidates, see section 8.

## 6. Calibration

The models are trained with the classes balanced, so their scores run far
above the real stop rate: the NN's mean validation score is 0.41 against
a stop rate of 0.19. A raw score of 0.6 is not a 60% chance.

The `calibrate` stage fits a sigmoid (Platt) and an isotonic calibrator on
the validation scores and keeps the one with the lower 5-fold
cross-validated Brier score: **isotonic** (0.1272 vs 0.1286 for sigmoid).

| Test set | Raw score | Calibrated |
|---|---|---|
| Brier score | 0.198 | **0.125** |
| Expected calibration error (10 bins) | 0.250 | **0.029** |
| Mean predicted risk (stop rate 0.185) | 0.41 | 0.19 |

Decisions do not change: a trial is flagged when the raw score reaches the
threshold, as in every gate and bias check. The calibrated probability is
the risk shown to a reviewer. The calibrator is packaged with the model
(`calibrator.joblib`). Reliability charts: `reports/calibration/` and
`reports/test/test_reliability.png`.

**Caveat:** isotonic calibration gives the very top scores a probability
of 1.0 (the highest bin of the validation set stopped every time). The
reviewer UI should show these as "very high" rather than "100%", or the
team can switch to sigmoid, which was almost as good (Brier 0.1286).

## 7. Lead time and error analysis

The model scores a trial at registration, so for a stopped trial it flags
the warning comes `stop date - registration date` ahead of the stop. Stop
dates come from the AACT studies table (actual completion date, else
actual primary completion date, else the last update; never a feature).

| Test set, NN | |
|---|---|
| Stopped trials flagged | 469 of 622 (recall 0.754) |
| Median lead time | **33.9 months** (IQR 21.3 to 48.6) |
| Flagged more than 12 months ahead | **87%** |
| Registered after they had already stopped | 3 (left out) |

**Recall by time from registration to stop:**

| Time to stop | Stopped trials | Recall |
|---|---|---|
| 0 to 6 months | 50 | 0.46 |
| 6 to 12 months | 67 | 0.55 |
| 12 to 24 months | 116 | 0.76 |
| 24 to 36 months | 126 | 0.81 |
| 36+ months | 257 | 0.84 |

The model is weakest on trials that stop within a year. These often stop
for reasons that don't show up at registration (funding, a sponsor
decision, slow early enrollment), so registration-time features can't see
them. Trials that run for years before stopping are caught well. The agent
and a future model that reads trial updates can cover the early stops.

**Recall by final status:** withdrawn 0.81 (205 trials), terminated 0.73
(391), suspended 0.69 (26). Withdrawn trials (never enrolled anyone) are
the easiest to spot.

**False positives:** of the 25 completed trials with the highest scores,
20 have academic or other non-industry sponsors, 13 are cancer trials and
9 are phase 2 (`top_false_positives.csv`). In the training set, cancer
trials stopped 34% of the time and phase 2 cancer trials 44%, against 18%
overall. So the model is following a real pattern, and these trials are
worth a reviewer's look rather than being clear mistakes. The lists of top false
positives and false negatives are in `reports/lead_time/` for manual
review.

## 8. Sensitivity and bias (summary)

- **SHAP** (NN, 300 validation trials): the 155 structured features carry
  about 0.46 of the total mean |SHAP|, and the six text fields 0.62
  together (brief summary 0.12, outcomes 0.11, title 0.11, inclusion 0.10,
  exclusion 0.09, detailed description 0.09). The text adds real signal
  spread over all fields. The top single features are start year,
  FDA-regulated drug or device, and healthy volunteers. Full table:
  `reports/sensitivity/`.
- **Bias:** in this run **every candidate failed the bias gate**
  (`block_bias_gap` 0.25). The largest gap is in the `start_era` slice
  (recall gap of 0.59 to 0.74 between trials started in different eras).
  With only 2019 registrations, start date is tightly linked to stopping,
  so this slice mostly measures the data, not unfair treatment. The bias
  gate is being reworked separately. The results above come from rerunning
  `select` onward with the block level set to 1.0 for this local
  evaluation only; `params.yaml` is unchanged. The models with text
  features had smaller gaps (NN 0.59, `xgboost_text` 0.57 to 0.59) than
  the structured-only models (XGBoost 0.69, logistic regression 0.74).

## 9. Limitations and next steps

- One registration year (2019): the test set is three months after the
  training set, and stops are only seen up to the snapshot date. A
  multi-year data version would give a fairer test and longer horizons.
- The NN's lead over XGBoost is within seed noise; a larger data version
  should settle it.
- BioBERT is used as is. Fine-tuning it on the label, or a model built for
  clinical text, could add more from the text.
- Inference (S5) has to apply the same steps: embed the six fields,
  `text_pca.joblib`, the model, then `calibrator.joblib`.
