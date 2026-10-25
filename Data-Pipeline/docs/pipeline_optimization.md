# Pipeline flow optimization (Airflow Gantt chart)

Rubric item 9 asks to find bottlenecks with Airflow's Gantt chart and to
speed up slow parts by parallelizing or improving them. This note records
what was measured, what was changed and the result.

## How it was measured

- Same DAG, same machine (Apple laptop), Airflow 2.10.5 with the
  LocalExecutor on PostgreSQL, so tasks can run at the same time.
- Placeholder snapshot with 50,000 trials (`acquire.sample_trials: 50000`),
  structured features from `src/features/structured_features.py`.
- Each run used a different snapshot and seed, so DVC re-ran every stage
  (an unchanged stage is skipped and would make a run look faster than it is).
- Baseline: the default pool limited to 1 slot, so tasks run one after
  another, the way the first version of the DAG ran them
  (`acquire >> preprocess >> schema_stats >> validate >> publish`, with bias
  and anomalies added in a line).
- Optimized: the same DAG with parallel slots allowed.

## Bottleneck found

The baseline Gantt chart showed every task waiting for the one before it,
although most of them do not need each other's output:

| Task | Needs | Sequential run (s) |
|---|---|---|
| acquire | nothing | 5.5 |
| preprocess | raw snapshot | 4.6 |
| schema_stats | features | 3.3 |
| bias | raw snapshot only | 3.9 |
| anomalies | features | 2.8 |
| validate | features, schema, statistics | 4.8 |
| publish | everything | 0.6 |
| **whole run** | | **30.3** |

![Sequential baseline](images/gantt_sequential_baseline.png)

## Change

The DAG now follows the real data dependencies (`dags/ctt_data_pipeline.py`):

```
               +--> preprocess --+--> schema_stats --> validate --+
    acquire ---+                 +--> anomalies -------------------+--> publish
               +--> bias ------------------------------------------+
```

- `bias` reads only the raw snapshot, so it runs next to `preprocess`.
- `schema_stats` and `anomalies` both read only the features, so they run
  next to each other.
- Parallel tasks are safe with DVC: locally DVC serializes its own writes to
  `dvc.lock`; on Cloud Composer each task runs in its own pod, so
  `scripts/run_stage.py` merges its stage into the shared `dvc.lock` in GCS
  with a conditional write that is retried when another task wrote first
  (tested in `tests/test_orchestration.py`).

## Result

| Run | Duration (s) |
|---|---|
| Sequential baseline | 30.3 |
| Parallel branches | 23.1 |

The run is **24% shorter** on the 50,000-trial sample. The remaining
critical path is `acquire -> preprocess -> schema_stats -> validate ->
publish`.

![Parallel branches](images/gantt_parallel_branches.png)

## Notes for the real AACT data

- On the full AACT export (about 500,000 trials) `bias`, `preprocess` and
  `validate` take minutes rather than seconds, so running `bias` next to
  `preprocess` saves more in absolute time.
- On this sample about 2 to 3 seconds of every task is fixed start-up cost
  (importing pandas and Great Expectations, DVC hashing its inputs), which
  is why the gain is capped here. It shrinks in relative terms on real data.
- A second run on the same snapshot is much faster again: DVC skips every
  stage whose code, data and parameters did not change, and `--pull`
  restores their outputs from the remote instead of recomputing them.

## Reproduce

```bash
# PostgreSQL for Airflow's metadata (the LocalExecutor needs it)
docker run -d --name airflow-pg -e POSTGRES_USER=airflow \
  -e POSTGRES_PASSWORD=airflow -e POSTGRES_DB=airflow \
  -p 127.0.0.1:5433:5432 postgres:16-alpine
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@127.0.0.1:5433/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags CTT_RUNNER=local CTT_PYTHON=$PWD/.venv/bin/python
airflow db migrate && airflow standalone
# baseline: one task at a time; optimized: parallel slots
airflow pools set default_pool 1 "sequential"   # then trigger a run
airflow pools set default_pool 16 "parallel"    # then trigger another run
```

On macOS, Airflow 2.10's LocalExecutor workers need the `fork` start method
(the default `spawn` fails with "'QueuedLocalWorker' object has no attribute
'wrapper'"); Linux and Cloud Composer use `fork` already.
