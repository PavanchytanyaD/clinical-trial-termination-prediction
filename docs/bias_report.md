# Bias Detection and Mitigation Report

Covers guideline section 3 (Data Bias Detection Using Data Slicing).
Parts 1–2 (detection and slicing): Danush. Parts 3–4 (mitigation and documentation): Atharva.

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
- Effect on model accuracy: to be measured once a model exists.

## 6. Limitations and next steps

- **Model-level check:** once predictions exist, set `model.predictions_path` and compare
  per-group PR-AUC and recall. If a group (e.g. non-US trials) is served worse, apply
  per-group thresholds (Fairlearn `ThresholdOptimizer`) or train with `trial_weights.csv`
  and compare.
- **Missing start date:** trials with no start date stop at 24.1%, well above average.
  The modeling team should check that start-date availability does not leak outcome
  information (for example, withdrawn trials that never started).
- **Tiny groups** (`AMBIG`, `disorders of environmental origin`) are too small for the model
  to learn from and are reported, not mitigated.
- **Re-run** on Nithish's cleaned data once `input.source` switches from `aact` to `file`.