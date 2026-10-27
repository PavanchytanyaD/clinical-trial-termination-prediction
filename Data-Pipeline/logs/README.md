# Logs

Every pipeline step writes its logs here while it runs. Nothing else in this
folder is committed.

- `<step>.log`: the step's own log (for example `validate.log`).
- `run_stage_<step>.log`: the stage runner: DVC commands, files pulled and
  pushed, `dvc.lock` merges, and any error with its traceback.
- `runs/<run_id>.jsonl`: one JSON record per step of an Airflow run: start
  and end time, duration, status, error, git commit, metrics (rows, label
  counts, checks passed) and an md5 fingerprint of each output.
- `versions/<snapshot>__<run_id>.json`: the data-version record written by
  `publish`.

Each task runs in a fresh copy of the code, so after the step these files
are uploaded to the run's state folder, where they are kept:

- Google Cloud: `gs://<bucket>/data-pipeline/state/runs/<run_id>/<step>/logs/`
- Local Docker Airflow: `local/output/state/runs/<run_id>/<step>/logs/`

The same log lines are in the Airflow task log (and in Cloud Logging on
Composer). See "6. Tracking and logging" in `../README.md`.
