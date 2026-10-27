# Baseline schema

Every new snapshot is validated against a baseline frozen from a trusted
snapshot:

- `expectation_suite.json`: the Great Expectations suite (schema) of that snapshot
- `statistics.json`: its statistics, used for drift checks
- `frozen_from.json`: which snapshot, when, and why

The baseline lives in the run-state folder: in GCS on Google Cloud
(`gs://ctt-mlops-project-data/data-pipeline/state/baseline/`), or in
`local/output/state/baseline/` on a local run. Every pipeline task downloads
it into this folder before running, so `validate` compares each snapshot
with it. Freeze one after a run on a snapshot you trust:

```bash
REASON="first real AACT snapshot" ./composer/deploy.sh freeze   # Google Cloud
REASON="first real AACT snapshot" ./local/pipeline.sh freeze     # local
```

Until a baseline exists, `validate` checks each snapshot against its own
generated suite and says so in the report.
