"""Airflow DAG: monthly data pipeline for clinical-trial termination.

               +--> preprocess --+--> schema_stats --> validate --+
               |                 +--> anomalies -------------------+
    acquire ---+--> bias ------------------------------------------+--> publish
               +--> text --> embeddings ---------------------------+

Each task runs one DVC stage (scripts/run_stage.py), so DVC versions every
output and skips stages whose inputs did not change; Airflow schedules the
run, orders the steps, retries failed steps and emails on failure.
`embeddings` computes the BioBERT text embeddings in a Vertex AI GPU job and
waits for it; DVC skips it when the text did not change.
Independent branches run in parallel (bias needs only the raw snapshot;
schema_stats and anomalies both need only the features), which shortens the
run; see docs/pipeline_optimization.md.
`validate` and `anomalies` exit non-zero on critical data problems, which
fails the task and stops the run before `publish` records a bad data
version.

It runs in one of two places (CTT_RUNNER):

  composer  (default) Cloud Composer, which only orchestrates: each task is
            a Kubernetes pod started from the pipeline image in Artifact
            Registry (docker/Dockerfile, built and pushed by CI), with its
            run state in GCS and its data in the DVC remote in GCS.
  local     the Docker Compose Airflow in local/: each task runs
            local/run_task.sh, which copies the code to a temporary folder
            (as fresh as a pod); run state and the DVC remote are folders on
            the machine, and BioBERT runs on its CPU. No cloud account.

A second DAG, ctt_freeze_baseline, makes the latest run's schema and
statistics the validation baseline: {"reason": "first real AACT snapshot"}.

Configuration (environment variables of the Composer environment, set by
composer/deploy.sh, or of the local containers, set in local/):

  CTT_RUNNER        composer or local
  CTT_IMAGE         pipeline image in Artifact Registry
  CTT_STATE_URI     shared run state (default gs://ctt-mlops-project-data/
                    data-pipeline/state; a folder for the local runner)
  CTT_DVC_REMOTE    DVC remote, if not the one in .dvc/config
  CTT_GCP_PROJECT, CTT_REGION, CTT_BUCKET, CTT_GPU_IMAGE,
  CTT_SERVICE_ACCOUNT
                    where the Vertex AI embeddings job runs
  CTT_EMBEDDINGS    vertex (default) or local (BioBERT on the CPU)
  CTT_ALERT_EMAILS  comma-separated failure-email recipients
  CTT_SOURCE        default acquire.source (sample, zip or db)
  CTT_ZIP           default acquire.zip, for source zip
  CTT_SMTP_SECRET   Kubernetes secret with SMTP_* / ALERT_* / AACT_* keys
  CTT_NAMESPACE     pod namespace (Composer 3: composer-user-workloads)

Trigger with a config to pick the snapshot or source for one run:
    {"snapshot": "20261001", "source": "zip", "zip": "gs://.../export.zip"}
and, for the test source, its size: {"source": "sample", "sample_trials": 50000}
"""

import json
import logging
import os
import shlex
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

log = logging.getLogger(__name__)

RUNNER = os.environ.get("CTT_RUNNER", "composer")
IMAGE = os.environ.get(
    "CTT_IMAGE",
    "us-central1-docker.pkg.dev/ctt-mlops-project/ctt-images/data-pipeline:latest")
STATE_URI = os.environ.get("CTT_STATE_URI",
                           "gs://ctt-mlops-project-data/data-pipeline/state")
# local runner: the script each task runs (next to this repository's dags/)
RUN_TASK = Path(__file__).resolve().parents[1] / "local" / "run_task.sh"
# passed to every task as PIPELINE_<name> when set (see the docstring)
DEPLOYMENT = ["DVC_REMOTE", "GCP_PROJECT", "REGION", "BUCKET", "GPU_IMAGE",
              "SERVICE_ACCOUNT", "EMBEDDINGS"]
ALERT_EMAILS = [e.strip() for e in
                os.environ.get("CTT_ALERT_EMAILS", "").split(",") if e.strip()]
NAMESPACE = os.environ.get("CTT_NAMESPACE", "composer-user-workloads")
SMTP_SECRET = os.environ.get("CTT_SMTP_SECRET", "")

# Per-run parameters: trigger config first, then environment defaults.
# Templates use double quotes only, so shell-quoting them keeps them intact.
SNAPSHOT = ('{{ dag_run.conf.get("snapshot") '
            'or data_interval_end.strftime("%Y%m%d") }}')
SOURCE = ('{{ dag_run.conf.get("source") or '
          + json.dumps(os.environ.get("CTT_SOURCE", "sample")) + " }}")
ZIP = ('{{ dag_run.conf.get("zip") or '
       + json.dumps(os.environ.get("CTT_ZIP", "")) + " }}")
# empty = keep params.yaml's value
SAMPLE_TRIALS = '{{ dag_run.conf.get("sample_trials") or "" }}'
REASON = '{{ dag_run.conf.get("reason") or "trusted snapshot" }}'


def log_failure(context):
    """Write one clear line per failed task (Cloud Logging picks it up)."""
    ti = context["task_instance"]
    log.error("ctt_data_pipeline task failed: dag_run=%s task=%s try=%s "
              "log=%s", context["run_id"], ti.task_id, ti.try_number,
              ti.log_url)


def step_env():
    env = {"PIPELINE_RUN_ID": "{{ run_id }}",
           "PIPELINE_STATE_URI": STATE_URI,
           # run_stage.py emails a task failure once, on the last attempt;
           # the stage itself then does not email it a second time
           "PIPELINE_ORCHESTRATED": "1",
           "PIPELINE_TRY": "{{ ti.try_number }}",
           "PIPELINE_MAX_TRIES": "{{ task.retries + 1 }}"}
    for name in DEPLOYMENT:
        if os.environ.get(f"CTT_{name}"):
            env[f"PIPELINE_{name}"] = os.environ[f"CTT_{name}"]
    return env


def step(stage, params=None, extra_args=(), timeout=timedelta(hours=2)):
    """One pipeline stage: a pod running the pipeline image (composer) or
    local/run_task.sh (local).

    params are params.yaml overrides ({"acquire.snapshot": "..."}); values
    may be Jinja templates.
    """
    args = [stage]
    for key, value in (params or {}).items():
        args += ["--param", f"{key}={value}"]
    args += list(extra_args)
    if RUNNER == "local":
        from airflow.operators.bash import BashOperator
        return BashOperator(
            task_id=stage,
            bash_command=shlex.join(["bash", str(RUN_TASK), *args]),
            # the container's environment (AACT and SMTP settings) plus ours
            env=step_env(),
            append_env=True,
            execution_timeout=timeout,
        )

    from airflow.providers.cncf.kubernetes.operators.pod import \
        KubernetesPodOperator
    from airflow.providers.cncf.kubernetes.secret import Secret
    from kubernetes.client import models as k8s
    secrets = [Secret("env", None, SMTP_SECRET)] if SMTP_SECRET else []
    return KubernetesPodOperator(
        task_id=stage,
        name=f"ctt-{stage.replace('_', '-')}",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        cmds=["python", "-m", "scripts.run_stage"],
        arguments=args,
        env_vars=step_env(),
        secrets=secrets,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "4Gi"},
            limits={"cpu": "2", "memory": "8Gi"}),
        # Composer 3 runs user pods through this connection and config file
        kubernetes_conn_id="kubernetes_default",
        config_file="/home/airflow/composer_kube_config",
        get_logs=True,
        on_finish_action="delete_pod",
        execution_timeout=timeout,
    )


default_args = {
    "owner": "ctt-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email": ALERT_EMAILS,
    "email_on_failure": bool(ALERT_EMAILS),
    "email_on_retry": False,
    "on_failure_callback": log_failure,
}

with DAG(
    dag_id="ctt_data_pipeline",
    description="AACT snapshot -> features, bias report -> statistics, "
                "anomalies, validation -> DVC data version",
    # AACT publishes a monthly export; run on the 5th at 06:00 UTC
    schedule="0 6 5 * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    # runs must not overlap: a run's stages share dvc.lock and params.yaml
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=12),
    default_args=default_args,
    tags=["ctt", "data-pipeline", "dvc"],
    doc_md=__doc__,
) as dag:
    acquire = step("acquire", {"acquire.snapshot": SNAPSHOT,
                               "acquire.source": SOURCE,
                               "acquire.zip": ZIP,
                               "acquire.sample_trials": SAMPLE_TRIALS})
    preprocess = step("preprocess")
    bias = step("bias")
    text = step("text")
    embeddings = step("embeddings", timeout=timedelta(hours=10))
    schema_stats = step("schema_stats")
    anomalies = step("anomalies", timeout=timedelta(minutes=45))
    validate = step("validate", timeout=timedelta(minutes=45))
    # publish runs only if every check passed (default trigger rule
    # all_success), so no bad data version is recorded
    publish = step("publish", timeout=timedelta(minutes=30))

    acquire >> [preprocess, bias, text]
    preprocess >> [schema_stats, anomalies]
    schema_stats >> validate
    text >> embeddings
    [validate, anomalies, bias, embeddings] >> publish


with DAG(
    dag_id="ctt_freeze_baseline",
    description="Make the latest run's schema and statistics the baseline",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args={**default_args, "retries": 0},
    tags=["ctt", "data-pipeline"],
) as freeze_dag:
    step("freeze", extra_args=["--reason", REASON],
         timeout=timedelta(minutes=30))
