# Baseline schema

Every new snapshot is validated against the files in this folder:

- `expectation_suite.json`: the Great Expectations suite (schema) of a trusted snapshot
- `statistics.json`: that snapshot's statistics, used for drift checks
- `frozen_from.json`: which snapshot, when, and why

Create or replace them after a run on a snapshot you trust, then commit them:

```bash
python -m scripts.schema_stats --freeze --reason "first real AACT snapshot"
git add schema/baseline && git commit -m "Freeze baseline schema"
```

Until a baseline exists, `validate` checks each snapshot against its own
generated suite and says so in the report.
