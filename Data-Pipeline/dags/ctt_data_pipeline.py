"""Airflow DAG: monthly data pipeline for clinical-trial termination.

    acquire -> preprocess -> schema_stats -> validate -> publish

Each task runs one DVC stage (scripts/run_stage.py), so DVC versions every
output and skips stages whose inputs did not change; Airflow schedules the
run, orders the steps, retries failed steps and emails on failure.
`validate` exits non-zero on critical data problems, which fails the task and
stops the run before `publish` records a bad data version.

Configuration (environment variables of the Composer environment):

  CTT_RUNNER        k8s (Cloud Composer, default) or local
  CTT_IMAGE         pipeline image in Artifact Registry (k8s runner)
  CTT_STATE_URI     gs://<bucket>/data-pipeline/state, shared run state
  CTT_ALERT_EMAILS  comma-separated failure-email recipients
  CTT_SOURCE        default acquire.source (sample, zip or db)
  CTT_ZIP           default acquire.zip, for source zip
  CTT_SMTP_SECRET   Kubernetes secret with SMTP_* / ALERT_* / AACT_* keys
  CTT_NAMESPACE     pod namespace (Composer 3: composer-user-workloads)
  CTT_PIPELINE_HOME Data-Pipeline folder (local runner)
  CTT_PYTHON        python with requirements.txt installed (local runner)

Trigger with a config to pick the snapshot or source for one run:
    {"snapshot": "20261001", "source": "zip", "zip": "gs://.../export.zip"}
"""

import json
import logging
import os
import shlex
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

log = logging.getLogger(__name__)

RUNNER = os.environ.get("CTT_RUNNER", "k8s")
IMAGE = os.environ.get(
    "CTT_IMAGE",
    "us-central1-docker.pkg.dev/ctt-mlops/ctt/data-pipeline:latest")
STATE_URI = os.environ.get("CTT_STATE_URI", "")
ALERT_EMAILS = [e.strip() for e in
                os.environ.get("CTT_ALERT_EMAILS", "").split(",") if e.strip()]
NAMESPACE = os.environ.get("CTT_NAMESPACE", "composer-user-workloads")
SMTP_SECRET = os.environ.get("CTT_SMTP_SECRET", "")
PIPELINE_HOME = os.environ.get(
    "CTT_PIPELINE_HOME",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON = os.environ.get("CTT_PYTHON", "python")

# Per-run parameters: trigger config first, then environment defaults.
# Templates use double quotes only, so shell-quoting them keeps them intact.
SNAPSHOT = ('{{ dag_run.conf.get("snapshot") '
            'or data_interval_end.strftime("%Y%m%d") }}')
SOURCE = ('{{ dag_run.conf.get("source") or '
          + json.dumps(os.environ.get("CTT_SOURCE", "sample")) + " }}")
ZIP = ('{{ dag_run.conf.get("zip") or '
       + json.dumps(os.environ.get("CTT_ZIP", "")) + " }}")

def log_failure(context):
    """Write one clear line per failed task (Cloud Logging picks it up)."""
    ti = context["task_instance"]
    log.error("ctt_data_pipeline task failed: dag_run=%s task=%s try=%s "
              "log=%s", context["run_id"], ti.task_id, ti.try_number,
              ti.log_url)


def step_env():
    env = {"PIPELINE_RUN_ID": "{{ run_id }}",
           # Airflow sends its own failure email; steps only email
           # validation results
           "PIPELINE_ORCHESTRATED": "1"}
    if STATE_URI:
        env["PIPELINE_STATE_URI"] = STATE_URI
    return env


def step(stage, params=None, timeout=timedelta(hours=2)):
    """One pipeline stage as a task, on the configured runner.

    params are params.yaml overrides ({"acquire.snapshot": "..."}); values
    may be Jinja templates.
    """
    args = [stage]
    for key, value in (params or {}).items():
        args += ["--param", f"{key}={value}"]
    if RUNNER == "local":
        cmd = " ".join(["cd", shlex.quote(PIPELINE_HOME), "&&", PYTHON, "-m",
                        "scripts.run_stage", *map(shlex.quote, args)])
        return BashOperator(task_id=stage, bash_command=cmd, env=step_env(),
                            append_env=True, execution_timeout=timeout)

    from airflow.providers.cncf.kubernetes.operators.pod import \
        KubernetesPodOperator
    from kubernetes.client import models as k8s

    secrets = []
    if SMTP_SECRET:
        from airflow.providers.cncf.kubernetes.secret import Secret
        secrets = [Secret("env", None, SMTP_SECRET)]
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
    description="AACT snapshot -> features -> schema/statistics -> "
                "validation -> DVC data version",
    # AACT publishes a monthly export; run on the 5th at 06:00 UTC
    schedule="0 6 5 * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,  # stages share dvc.lock, so runs must not overlap
    dagrun_timeout=timedelta(hours=8),
    default_args=default_args,
    tags=["ctt", "data-pipeline", "dvc"],
    doc_md=__doc__,
) as dag:
    acquire = step("acquire", {"acquire.snapshot": SNAPSHOT,
                               "acquire.source": SOURCE,
                               "acquire.zip": ZIP})
    preprocess = step("preprocess")
    schema_stats = step("schema_stats")
    validate = step("validate", timeout=timedelta(minutes=45))
    # validate failing stops here, so no bad data version is recorded
    publish = step("publish", timeout=timedelta(minutes=30))

    acquire >> preprocess >> schema_stats >> validate >> publish
