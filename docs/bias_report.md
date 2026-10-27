# Bias Detection and Mitigation Report

Covers guideline section 3 (Data Bias Detection Using Data Slicing).
Parts 1–2 (detection and slicing): Danush. Parts 3–4 (mitigation and documentation): Atharva.
Section 7 (model-level check with a baseline model): Pavanchytanya.

Bias is checked twice in every pipeline run:

1. **Data level** (stage `bias`, sections 1–6): how many finished trials each group has and how
   often they stop; small groups are reweighted.
2. **Model level** (stage `bias_model`, section 7): a baseline model is trained and evaluated per
   group with Fairlearn, then mitigated two ways (training with the section 3 weights, and one
   decision threshold per group).

Source: full AACT pull, run with `input.limit: null`.
Outputs: `reports/bias/data_summary.json`, `reports/bias/data_slices.csv`,
`reports/bias/mitigation/mitigation_summary.json`, `reports/bias/mitigation/mitigation_before_after.csv`.

## 1. Data and slices

| | Count |
|---|---|
| Trials pulled | 608,221 |
| Finished (label known) | 385,942 (63.5%) |
| Unfinished (censored, no label) | 222,279 (36.5%) |
| Overall stopped rate (finished trials) | 13.8% |

Our data is trial-level, so there are no patient demographics. We slice on trial
attributes, using eligibility gender and age group as the closest demographic proxies.

Slices: phase, study type, sponsor class, region, therapeutic area, gender, age group, start era.

Flag rules (from `configs/bias_config.yaml`):
- **Too small:** under 1% of finished trials (`min_share: 0.01`)
- **Large gap:** stopped rate more than 10 points from the overall 13.8% (`max_rate_gap: 0.10`)

## 2. Bias found

### 2.1 Representation bias (groups too small)

| Slice | Group | Share | Stopped rate |
|---|---|---|---|
| phase | EARLY_PHASE1 | 0.91% | 22.9% |
| sponsor_class | FED | 0.99% | 12.4% |
| sponsor_class | NETWORK | 0.91% | 15.8% |
| sponsor_class | INDIV | 0.10% | 15.5% |
| sponsor_class | AMBIG | <0.01% (3 trials) | 33.3% |
| therapeutic_area | occupational diseases | 0.12% | 7.4% |
| therapeutic_area | disorders of environmental origin | 0.01% (21 trials) | 4.8% |
| study_type | EXPANDED_ACCESS | 0% finished | n/a |

Groups named `UNKNOWN` (missing values) were also flagged in several slices. They are
data-quality issues, not real subgroups, so they are not mitigated.

`EXPANDED_ACCESS` has no finished trials, because expanded-access records do not use
the completed/terminated statuses. It cannot get a label and is out of scope for the model.

### 2.2 Stopped-rate gaps

| Slice | Group | Stopped rate | Gap vs 13.8% |
|---|---|---|---|
| phase | PHASE1/PHASE2 | 27.6% | +13.8 pts |
| start_era | UNKNOWN (no start date) | 24.1% | +10.4 pts |
| sponsor_class | AMBIG | 33.3% | +19.5 pts (3 trials, noise) |
| region | US_ONLY vs NON_US_ONLY | 19.7% vs 9.0% | not flagged against the overall rate, but a 2x difference between regions |

### 2.3 Censoring bias

The share of unfinished trials rises from 9.5% (2000–2009) to 53.6% (2018 and later).
Recent trials are therefore under-represented among labeled trials, so the data
reflects older trial practice more than current practice.

## 3. Mitigation

| Problem | What we did | Why |
|---|---|---|
| Small groups (≥100 trials) | Reweighting: each group lifted to a 1% share, weights capped at 10x | Gives small groups enough influence without inventing or duplicating data |
| Small groups (<100 trials) | Not boosted, listed as limitations (`min_group_size: 100`) | Weighting a handful of trials up only amplifies noise |
| `UNKNOWN` groups | Not boosted | Missing values, not real subgroups |
| Stopped-rate gaps (phase, region) | Not changed | Likely real differences in how trials behave. Forcing equal rates would teach the model something false. Checked again at model level (section 6) |
| Censoring of recent trials | Not fixable by resampling; documented | The labels do not exist yet. The time-based split already evaluates on recent trials |

Method: `src/bias/mitigate.py`, `compute_weights()`. A group with share *s* below the
target *m* has its trials' weights multiplied by *m(1−s) / (s(1−m))*. This repeats until
every group reaches *m* or hits the cap, because boosting one slice slightly shifts the
others. Weights are then scaled to average 1. Output: `trial_weights.csv` (`nct_id, weight`),
for use as sample weights in model training.

## 4. Results (before vs after)

| Slice | Group | Trials | Share before | Share after | Mean weight |
|---|---|---|---|---|---|
| phase | EARLY_PHASE1 | 3,502 | 0.91% | 1.0% | 1.11 |
| sponsor_class | FED | 3,833 | 0.99% | 1.0% | 1.01 |
| sponsor_class | NETWORK | 3,511 | 0.91% | 1.0% | 1.10 |
| sponsor_class | INDIV | 393 | 0.10% | 0.998% | 9.80 |
| therapeutic_area | occupational diseases | 449 | 0.12% | 1.0% | 8.60 |

Not boosted (too few trials): `AMBIG` sponsor (3 trials) and
`disorders of environmental origin` (21 trials).

All five boosted groups reached the 1% target. `INDIV` stopped just short (0.998%) because
of the 10x cap, which is effectively at target.

## 5. Trade-offs

| | Before | After |
|---|---|---|
| Overall stopped rate | 13.80% | 13.77% |
| Effective sample size | 385,942 | 336,675 (87.2%) |
| Largest weight | 1.0 | 9.8 |

- The overall stopped rate barely moves, so reweighting does not distort the label balance.
- Uneven weights reduce the effective sample size by about 13%. Most of this cost comes from
  the two heavily boosted groups (`INDIV`, `occupational diseases`).
- Lowering `max_weight` to 5 would cut this cost but leave those two groups below 1%.
  We kept 10 as a balance between fair representation and information loss.
- Effect on model accuracy: measured with the baseline model in section 7. Reweighting leaves
  it unchanged (PR-AUC 0.486 before and after), because the boosted groups are a few percent of
  the trials.

## 6. Limitations and next steps

- **Model-level check:** done with a baseline model (section 7). Repeat it for the BioBERT
  model in the model development assignment: set `model.predictions_path` to its predictions.
- **Missing start date:** trials with no start date stop at 24.1%, well above average.
  The modeling team should check that start-date availability does not leak outcome
  information (for example, withdrawn trials that never started).
- **Tiny groups** (`AMBIG`, `disorders of environmental origin`) are too small for the model
  to learn from and are reported, not mitigated.
- **Re-run** on Nithish's cleaned data once `input.source` switches from `aact` to `file`.

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

**Outputs** (each run, in `Data-Pipeline/reports/bias_model/`; the 2019 run is copied to
[`docs/bias_model_2019/`](bias_model_2019/)): `bias_model.md` (this report's tables),
`model_slices.csv` (every group, every model), `model_gaps.csv`, `summary.json`.
Settings: `params.yaml` section `bias_model` (test share, gap limit, minimum stopped trials,
threshold slice) and `configs/bias_config.yaml` (slices, threshold 0.5, minimum group size 100).
