# Baseline model: performance per subgroup

Logistic regression on 148 structured features; trained on 13402 finished trials, tested on the 3351 most recently registered (threshold 0.5). Models: baseline; reweighted (trained with the bias stage's trial weights); group_thresholds (baseline with one threshold per region group, for equal recall).

## Overall (test set)

| Model | PR-AUC | ROC-AUC | Recall | Precision | Flagged |
|---|---|---|---|---|---|
| baseline | 0.486 | 0.774 | 0.781 | 0.299 | 0.483 |
| reweighted | 0.486 | 0.775 | 0.781 | 0.300 | 0.481 |
| group_thresholds | 0.457 | 0.756 | 0.794 | 0.282 | 0.520 |

Stopped rate in the test set: 0.185.

## Gap between the best and the worst group

Groups with at least 100 test trials and 10 stopped ones, without UNKNOWN; a recall or PR-AUC gap above 0.1 is flagged.

| Slice | Model | Groups | pr_auc | recall | precision | flagged_rate |
|---|---|---|---|---|---|---|
| phase | baseline | 5 | 0.212 | 0.221 | 0.067 | 0.452 |
| sponsor_class | baseline | 2 | 0.043 | 0.003 | 0.022 | 0.096 |
| region | baseline | 3 | 0.250 | 0.319 | 0.161 | 0.502 |
| therapeutic_area | baseline | 17 | 0.341 | 0.462 | 0.227 | 0.556 |
| gender | baseline | 3 | 0.102 | 0.086 | 0.036 | 0.210 |
| age_group | baseline | 3 | 0.100 | 0.144 | 0.042 | 0.130 |
| start_era | baseline | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| phase | reweighted | 5 | 0.221 | 0.221 | 0.063 | 0.455 |
| sponsor_class | reweighted | 2 | 0.043 | 0.003 | 0.023 | 0.096 |
| region | reweighted | 3 | 0.249 | 0.323 | 0.163 | 0.500 |
| therapeutic_area | reweighted | 17 | 0.340 | 0.490 | 0.222 | 0.562 |
| gender | reweighted | 3 | 0.102 | 0.066 | 0.028 | 0.207 |
| age_group | reweighted | 3 | 0.103 | 0.144 | 0.048 | 0.122 |
| start_era | reweighted | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| phase | group_thresholds | 5 | 0.160 | 0.182 | 0.067 | 0.477 |
| sponsor_class | group_thresholds | 2 | 0.066 | 0.001 | 0.002 | 0.055 |
| region | group_thresholds | 3 | 0.250 | 0.133 | 0.249 | 0.188 |
| therapeutic_area | group_thresholds | 17 | 0.299 | 0.417 | 0.212 | 0.553 |
| gender | group_thresholds | 3 | 0.124 | 0.121 | 0.059 | 0.233 |
| age_group | group_thresholds | 3 | 0.079 | 0.159 | 0.057 | 0.118 |
| start_era | group_thresholds | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

## Flagged

- baseline: phase pr_auc gap 0.212
- baseline: phase recall gap 0.221
- baseline: region pr_auc gap 0.250
- baseline: region recall gap 0.319
- baseline: therapeutic_area pr_auc gap 0.341
- baseline: therapeutic_area recall gap 0.462
- baseline: gender pr_auc gap 0.102
- baseline: age_group recall gap 0.144
- reweighted: phase pr_auc gap 0.221
- reweighted: phase recall gap 0.221
- reweighted: region pr_auc gap 0.249
- reweighted: region recall gap 0.323
- reweighted: therapeutic_area pr_auc gap 0.340
- reweighted: therapeutic_area recall gap 0.490
- reweighted: gender pr_auc gap 0.102
- reweighted: age_group pr_auc gap 0.103
- reweighted: age_group recall gap 0.144
- group_thresholds: phase pr_auc gap 0.160
- group_thresholds: phase recall gap 0.182
- group_thresholds: region pr_auc gap 0.250
- group_thresholds: region recall gap 0.133
- group_thresholds: therapeutic_area pr_auc gap 0.299
- group_thresholds: therapeutic_area recall gap 0.417
- group_thresholds: gender pr_auc gap 0.124
- group_thresholds: gender recall gap 0.121
- group_thresholds: age_group recall gap 0.159
