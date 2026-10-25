# Deploying to Cloud Composer

The DAG runs on Cloud Composer 3 in the `ctt-mlops` project (project ID
`ctt-mlops-project`). Each task runs the pipeline image in a Kubernetes pod,
so Composer itself needs no extra Python packages. Everyday runs use the
Compute Engine VM instead (`../gce/`); both share the same bucket.

**Cost:** a Composer environment is billed every hour it exists, even when no
DAG runs, and even the small size uses a large share of the course credits
within days. Create it for a demo or a monthly run, then delete it
(`./composer/deploy.sh delete`). The DAG and the data in GCS stay; creating
the environment again takes about 25 minutes.

## Prerequisites

- `gcloud` logged in with an account that can use `ctt-mlops-project` (Composer,
  Cloud Build, Artifact Registry and Storage admin roles), and
  `gcloud config set project ctt-mlops-project`.
- The bucket `gs://ctt-mlops-project-data` for the DVC data and the run
  state (`./composer/deploy.sh bucket` creates it if missing).
- The environment's service account needs `roles/storage.objectAdmin` on that
  bucket, so the pods can read and write data.

## Steps

From the `Data-Pipeline` folder:

```bash
export ALERT_EMAILS=<a@x.com,b@y.com>
./composer/deploy.sh bucket    # create the bucket if needed
./composer/deploy.sh image     # Cloud Build -> Artifact Registry
./composer/deploy.sh gpu-image # BioBERT image for the Vertex AI embeddings job
./composer/deploy.sh env       # create the environment (about 25 minutes)
./composer/deploy.sh config    # CTT_* environment variables for the DAG
./composer/deploy.sh secret    # SMTP / AACT settings from .env as a secret
./composer/deploy.sh dag       # upload the DAG
SNAPSHOT=20261001 SOURCE=sample ./composer/deploy.sh trigger
```

Rebuild the image (`image`) after every code change: the pods run the code
in the image, not the code in the repository. Builds run as `cicd-sa`
(Artifact Registry writer, log writer), not the Compute default account.
The `embeddings` task only submits and watches the Vertex AI GPU job, so it
runs in the normal pipeline pod.

## DAG settings (environment variables)

| Variable | Value |
|---|---|
| `CTT_RUNNER` | `k8s` |
| `CTT_IMAGE` | `us-central1-docker.pkg.dev/ctt-mlops-project/ctt-images/data-pipeline:latest` |
| `CTT_STATE_URI` | `gs://ctt-mlops-project-data/data-pipeline/state` |
| `CTT_ALERT_EMAILS` | failure-email recipients |
| `CTT_SMTP_SECRET` | `ctt-pipeline-secrets` (SMTP / ALERT / AACT keys) |
| `CTT_SOURCE` | default `acquire.source`: `sample`, `zip` or `db` |

Airflow's own failure emails also need an email backend on the environment
(Composer's SendGrid integration, or Airflow SMTP settings). Without it the
validation alerts sent from the pods still work, through the SMTP secret.

## What ends up in the bucket

```
gs://ctt-mlops-project-data/
|- dvc-store/                       DVC remote: every data version
|- data-pipeline/state/
   |- dvc.lock                      latest successful stage outputs
   |- baseline/                     validation baseline (./gce/vm.sh freeze)
   |- runs/<run_id>/params.yaml     parameters of that run
   |- runs/<run_id>/<stage>/logs/   step logs and run records
   |- runs/<run_id>/<stage>/reports/  statistics and validation reports
   |- versions/<snapshot>/dvc.lock  data version per snapshot
   |- versions/<snapshot>/<run_id>.json
```

The VM's runs write to the same folders, so a run on either host continues
from the latest data versions of both.
