# Bias Detection and Mitigation Report

Covers guideline section 3. Detection and slicing (parts 1 and 2) by Danush,
mitigation and documentation (parts 3 and 4) by Atharva.

## 1. Data and slices
- Source: AACT, <N> trials (<N finished>, rest unfinished and censored)
- Our data is trial-level, so there are no patient demographics. We slice on trial
  attributes, including eligibility gender and age group as the closest demographic proxies.
- Slices: phase, study type, sponsor class, gender, therapeutic area, age group, region, start era

## 2. Bias found
| Slice | Group | Share | Stopped rate | Problem |
|---|---|---|---|---|
| sponsor_class | <group> | <x%> | <x%> | Too small (under 1%) |
| region | US_ONLY vs NON_US_ONLY | | 19.7% vs 9.0% | Large gap |
| phase | PHASE1/PHASE2 | | 27.6% | Large gap |
| start_era | 2018 and later | | | 53.6% unfinished (censoring) |

## 3. Mitigation
| Problem | What we did | Why |
|---|---|---|
| Small groups | Reweighting (`method: reweight`), capped at 10x | Gives small groups enough influence without inventing data |
| Stopped-rate gaps (region, phase) | Not forced equal | Likely real differences in how trials behave; forcing equal rates would teach the model something false. Re-checked at model level once a model exists |
| Censoring of recent trials | Documented; time-based split already used | Can't be fixed by resampling, since the labels don't exist yet |

## 4. Results (before vs after)
From `mitigation_before_after.csv` and `mitigation_summary.json`.
| Group | Share before | Share after | Stopped rate before | Stopped rate after |
|---|---|---|---|---|
| | | | | |

## 5. Trade-offs
- Effective sample size: <N> → <ESS> (uneven weights mean less information per trial)
- Overall stopped rate: <before> → <after>
- Model-level effect: to be filled in once predictions exist (PR-AUC overall and per group)

## 6. Limitations and next steps
- Model-level mitigation (per-group thresholds with Fairlearn `ThresholdOptimizer`) once a model exists
- Re-run on Nithish's cleaned data when the source switches from AACT to file