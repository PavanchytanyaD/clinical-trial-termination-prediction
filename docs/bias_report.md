# Bias Detection and Mitigation Report

Covers guideline section 3 (Data Bias Detection Using Data Slicing).

**Data and run.** Every number in this report comes from one run of the pipeline's DVC stages
(`dvc repro bias_model`: `acquire` → `preprocess` → `bias` → `bias_model`, run ID
`realdata_2019`) on the real AACT database (snapshot of 2026-10-27), with the pipeline's
real-data settings: interventional trials registered in 2019. The run's raw output is copied,
unchanged, to [bias_report_evidence/](bias_report_evidence/).

Bias is checked twice in every pipeline run:

1. **Data level** (stage `bias`, sections 1–6): how many finished trials each group has and how
   often they stop; small groups are reweighted. Output: `Data-Pipeline/reports/bias/`.
2. **Model level** (stage `bias_model`, section 7): a baseline model is trained and evaluated per
   group with Fairlearn, then mitigated two ways (training with the section 3 weights, and one
   decision threshold per group). Output: `Data-Pipeline/reports/bias_model/`.

## 1. Data and slices

| | Count |
|---|---|
| Trials (interventional, registered in 2019) | 24,236 |
| Finished (label known) | 16,753 (69.1%) |
| Unfinished (censored, no label) | 7,483 (30.9%) |
| Overall stopped rate (finished trials) | 18.4% (3,089 stopped) |

Our data is trial-level, so there are no patient demographics. We slice on trial
attributes, using eligibility gender and age group as the closest demographic proxies.

Slices: phase, study type, sponsor class, region, therapeutic area, gender, age group, start era.
Study type has a single group here, because the run keeps interventional trials only.

Flag rules (from `configs/bias_config.yaml`):
- **Too small:** under 1% of finished trials (`min_share: 0.01`), here fewer than 168 trials
- **Large gap:** stopped rate more than 10 points from the overall 18.4% (`max_rate_gap: 0.10`)

## 2. Bias found

### 2.1 Representation bias (groups too small)

| Slice | Group | Finished trials | Share | Stopped rate |
|---|---|---|---|---|
| sponsor_class | NIH | 100 | 0.60% | 34.0% |
| sponsor_class | NETWORK | 80 | 0.48% | 16.3% |
| sponsor_class | INDIV | 8 | 0.05% | 12.5% |
| therapeutic_area | occupational diseases | 25 | 0.15% | 20.0% |
| start_era | 2000–2009 | 96 | 0.57% | 3.1% |
| start_era | before 2000 | 9 | 0.05% | 0.0% |

Groups named `UNKNOWN` (missing values) were also flagged in phase (16 trials), sponsor class
(3) and gender (10). They are data-quality issues, not real subgroups, so they are not mitigated.

### 2.2 Stopped-rate gaps

| Slice | Group | Stopped rate | Gap vs 18.4% |
|---|---|---|---|
| phase | PHASE1/PHASE2 | 37.6% | +19.1 pts |
| phase | PHASE2 | 31.8% | +13.4 pts |
| phase | PHASE2/PHASE3 | 29.3% | +10.8 pts |
| therapeutic_area | hemic and lymphatic diseases | 35.8% | +17.4 pts |
| therapeutic_area | neoplasms | 34.5% | +16.1 pts |
| sponsor_class | NIH | 34.0% | +15.6 pts (also too small) |
| region | US_AND_NON_US | 29.8% | +11.4 pts |
| region | UNKNOWN (no country listed) | 29.1% | +10.7 pts |
| start_era | 2010–2017 | 3.0% | −15.5 pts |
| region | US_ONLY vs NON_US_ONLY | 24.3% vs 12.1% | not flagged against the overall rate, but a 2x difference between regions |

Gender and age group show no flagged gaps (stopped rates 13.8%–18.9%).

### 2.3 Censoring and registration bias

- **Censoring:** 30.9% of the trials have not finished, so they have no label yet. The share is
  highest for long or public trials (Phase 2/3 42.7%, other government sponsors 45.0%, NIH
  43.5%), so the labeled data leans towards shorter trials.
- **Registration lag:** trials registered in 2019 that started before 2018 were registered late,
  often after they had already finished, and almost all of them completed (stopped rate 3.0% for
  2010–2017). Their low rate most likely reflects how they entered the registry, not how older
  trials behave.

## 3. Mitigation

| Problem | What we did | Why |
|---|---|---|
| Small groups (≥100 trials) | Reweighting: each group lifted to a 1% share, weights capped at 10x | Gives small groups enough influence without inventing or duplicating data |
| Small groups (<100 trials) | Not boosted, listed as limitations (`min_group_size: 100`) | Weighting a handful of trials up only amplifies noise |
| `UNKNOWN` groups | Not boosted | Missing values, not real subgroups |
| Stopped-rate gaps (phase, area, region) | Not changed | Likely real differences in how trials behave. Forcing equal rates would teach the model something false. Checked again at model level (section 7) |
| Censoring and registration lag | Not fixable by resampling; documented | The labels do not exist yet, and late registration is a property of the registry. The time-based split in section 7 evaluates on the most recent trials |

Method: `src/bias/mitigate.py`, `compute_weights()`. A group with share *s* below the
target *m* has its trials' weights multiplied by *m(1−s) / (s(1−m))*. This repeats until
every group reaches *m* or hits the cap, because boosting one slice slightly shifts the
others. Weights are then scaled to average 1. Output: `trial_weights.csv` (`nct_id, weight`),
for use as sample weights in model training.

## 4. Results (before vs after)

| Slice | Group | Trials | Share before | Share after | Mean weight |
|---|---|---|---|---|---|
| sponsor_class | NIH | 100 | 0.60% | 1.0% | 1.68 |

NIH is the only small group with at least 100 trials, so it is the only one boosted; it reached
the 1% target. Not boosted (too few trials): `INDIV` (8), `NETWORK` (80),
`occupational diseases` (25), start era `2000–2009` (96) and `before 2000` (9).

## 5. Trade-offs

| | Before | After |
|---|---|---|
| Overall stopped rate | 18.44% | 18.50% |
| Effective sample size | 16,753 | 16,707 (99.7%) |
| Largest weight | 1.0 | 1.68 |

- The overall stopped rate barely moves, so reweighting does not distort the label balance.
- Only 100 trials get a weight above 1, so almost no information is lost (effective sample
  size 99.7%).
- Effect on model accuracy: measured with the baseline model in section 7. Reweighting leaves
  it unchanged (PR-AUC 0.486 before and after), because the boosted group is 0.6% of the trials.
- On the full AACT data (about 600,000 trials) more groups pass the 100-trial minimum and get
  boosted; the same code and settings apply.

## 6. Limitations and next steps

- **Model-level check:** done with a baseline model (section 7). Repeat it for the BioBERT
  model in the model development assignment: set `model.predictions_path` to its predictions.
- **Too few trials to mitigate:** `INDIV`, `NETWORK`, `occupational diseases` and the trials
  that started before 2010 are reported, not mitigated. A larger registration window would give
  them enough trials.
- **Missing country:** trials with no country listed stop at 29.1%, and section 7 shows the
  model finds them almost perfectly. The modeling team should check that this does not leak
  the outcome (withdrawn trials lose their countries).
- **Late registration:** decide in the model development assignment whether late-registered
  trials (started before 2018) should be kept, because their outcome is mostly known when
  they are registered.

## 7. Model-level check (baseline model)

The data-level checks above cannot say whether a *model* serves every group equally well, so the
pipeline trains one and checks it (`Data-Pipeline/scripts/bias_model.py`, stage `bias_model`,
which runs in every pipeline run after `preprocess` and `bias`).

**Setup**

- **Data:** the real AACT data the pipeline uses, interventional trials registered in 2019:
  24,236 trials, 16,753 finished (3,089 stopped, 18.4%).
- **Model:** logistic regression on the 148 structured features (the columns `preprocess` marks
  as a leakage risk are left out), with balanced class weights.
- **Split by time:** trained on the 13,402 earliest-registered finished trials, tested on the
  3,351 most recent ones, as the real model will predict for new trials.
- **Per-group evaluation:** Fairlearn `MetricFrame` (`src/bias/model_slices.py`) on the same
  slices as sections 1–2. The gap of a metric is the best group minus the worst group, over
  groups with at least 100 test trials and 10 stopped ones. `UNKNOWN` groups are reported but
  not counted, as in section 2. A recall or PR-AUC gap above 0.10 is flagged in the logs.
- **Mitigation, two ways:** (a) `reweighted`: the same model trained with `trial_weights.csv`
  from section 3; (b) `group_thresholds`: the baseline with one decision threshold per region,
  chosen on the training trials so that every region gets the same recall as the model overall
  (equal opportunity; "adjusting decision thresholds" in the guidelines).

**Overall (test set, stopped rate 18.5%)**

| Model | PR-AUC | ROC-AUC | Recall | Precision | Trials flagged |
|---|---|---|---|---|---|
| baseline | 0.486 | 0.774 | 0.781 | 0.299 | 48.3% |
| reweighted | 0.486 | 0.775 | 0.781 | 0.300 | 48.1% |
| group_thresholds | 0.457 | 0.756 | 0.794 | 0.282 | 52.0% |

**Bias found: region.** The baseline misses far more stopped trials outside the US:

| Region | Test trials | Recall: baseline | Recall: group thresholds | Precision: baseline | Precision: group thresholds |
|---|---|---|---|---|---|
| NON_US_ONLY | 1,854 | 0.651 | 0.777 | 0.196 | 0.177 |
| US_ONLY | 1,114 | 0.829 | 0.776 | 0.347 | 0.362 |
| US_AND_NON_US | 202 | 0.970 | 0.909 | 0.358 | 0.426 |

The recall gap between regions falls from **0.32 to 0.13** with per-region thresholds
(non-US 0.431, US-only 0.540, both 0.633). Reweighting does not change it (0.32), because it only
lifts groups below 1% of the trials.

**Trade-offs.** Equal recall costs some overall ranking quality: PR-AUC 0.486 to 0.457, precision
0.299 to 0.282, and 4 points more trials flagged overall (non-US trials 38.5% to 51.0%, US-only
trials 61.3% to 55.1%). For a tool that flags at-risk trials for review, missing
non-US stopped trials is the worse error, so we recommend per-group thresholds for the final model,
with the trade-off checked again then.

**Other gaps (baseline, groups as defined above):**

| Slice | Groups | PR-AUC gap | Recall gap | Comment |
|---|---|---|---|---|
| phase | 5 | 0.21 | 0.22 | follows the stopped rate (Phase 2: 35%, Phase 4: 22%); PR-AUC depends on it |
| therapeutic_area | 17 | 0.34 | 0.46 | many small areas; noisy, re-check with the final model |
| age_group | 3 | 0.10 | 0.14 | small |
| gender | 3 | 0.10 | 0.09 | small |
| sponsor_class | 2 | 0.04 | 0.00 | none (industry vs other) |

**Leakage found.** Trials with no country listed (region `UNKNOWN`, 181 test trials) are predicted
almost perfectly (PR-AUC 0.99, precision 0.98), although every country and facility column is
already excluded as a leakage risk. Withdrawn trials lose their countries, and some other column
still gives them away (most likely a missing-value pattern). This is not bias but a leakage
warning: the modeling team should find that column before training the real model.

**Outputs** (each run, in `Data-Pipeline/reports/bias_model/`; this run's copy is in
[`docs/bias_report_evidence/bias_model/`](bias_report_evidence/bias_model/)): `bias_model.md` (this report's tables),
`model_slices.csv` (every group, every model), `model_gaps.csv`, `summary.json`.
Settings: `params.yaml` section `bias_model` (test share, gap limit, minimum stopped trials,
threshold slice) and `configs/bias_config.yaml` (slices, threshold 0.5, minimum group size 100).
