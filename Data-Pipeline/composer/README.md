# Cloud Composer

The pipeline runs on Cloud Composer 3 in the `ctt-mlops` project (project ID
`ctt-mlops-project`). Composer only orchestrates: each task is a Kubernetes
pod started from the pipeline image in Artifact Registry, which GitHub
Actions builds on every push (`../../.github/workflows/ci.yml`), and the
`embeddings` task hands BioBERT to a Vertex AI GPU job.

The commands below use our project. In your own project, `export
PROJECT=<your project id>` first: the bucket (`<project>-data`), the
service accounts, the images and every setting the pods get follow from it.
The full walk-through, with screenshots of our deployment, is in
[the root README](../../README.md#way-2-on-google-cloud). To run the
pipeline without any cloud account, use [../local/](../local/README.md).

**Cost:** a Composer environment is billed every hour it exists, even when no
DAG runs, and it cannot be paused. So it is created for a working session
and deleted afterwards. The data, the run state and a snapshot of Airflow's
own history stay in the bucket, and `up` restores them.

## Once per project

```bash
gcloud auth login && gcloud config set project ctt-mlops-project
./composer/deploy.sh setup
```

`setup` enables the APIs, creates the bucket and the service accounts if
they are missing, and gives them their roles. `pipeline-sa` is the identity
of Composer, its pods and the Vertex AI job (Composer worker, Storage,
Artifact Registry reader, Vertex AI, Logging). `cicd-sa` builds the images.
Composer's own service agent gets the role it needs to act as
`pipeline-sa`.

## A working session

From the `Data-Pipeline` folder:

```bash
export ALERT_EMAILS=<a@x.com,b@y.com>   # optional: failure emails
./composer/deploy.sh up        # create (about 25 min), restore history,
                               # settings, AACT/SMTP secret, upload the DAGs
./composer/deploy.sh trigger   # SNAPSHOT=20261101 SOURCE=db ... to choose
./composer/deploy.sh ui        # the Airflow web UI address
./composer/deploy.sh lock      # copy the run's dvc.lock here, then commit it
SCHEDULE="@weekly" ./composer/deploy.sh schedule   # change how often it runs
                               # (live; "default" = monthly, "none" = manual)
./composer/deploy.sh down      # snapshot Airflow's history, then delete
```

`up` is the single steps `env`, `secret`, `config` and `dag`; each can also
be run on its own. The images are built by CI; `./composer/deploy.sh image`
and `gpu-image` build them by hand if needed (as `cicd-sa`, through Cloud
Build).

## DAG settings (environment variables)

| Variable | Value |
|---|---|
| `CTT_IMAGE` | `us-central1-docker.pkg.dev/ctt-mlops-project/ctt-images/data-pipeline:latest` (CI pushes it) |
| `CTT_STATE_URI` | `gs://ctt-mlops-project-data/data-pipeline/state` |
| `CTT_DVC_REMOTE` | `gs://ctt-mlops-project-data/dvc-store` |
| `CTT_GCP_PROJECT`, `CTT_REGION`, `CTT_BUCKET` | where the Vertex AI job runs and stages its files |
| `CTT_GPU_IMAGE`, `CTT_SERVICE_ACCOUNT` | the BioBERT image and the identity of the Vertex AI job |
| `CTT_ALERT_EMAILS` | failure-email recipients |
| `CTT_SMTP_SECRET` | `ctt-pipeline-secrets` (SMTP / ALERT / AACT keys), empty if `.env` has none |
| `CTT_SOURCE` | default `acquire.source`: `sample`, `zip` or `db` |

Airflow's own failure emails also need an email backend on the environment
(Composer's SendGrid integration, or Airflow SMTP settings). Without it the
validation alerts sent from the pods still work, through the SMTP secret.

## What ends up in the bucket

```
gs://ctt-mlops-project-data/
|- dvc-store/                       DVC remote: every data version
|- composer-snapshots/              Airflow's history, saved by down
|- data-pipeline/state/
   |- dvc.lock                      latest successful stage outputs
   |- baseline/                     validation baseline (deploy.sh freeze)
   |- runs/<run_id>/params.yaml     parameters of that run
   |- runs/<run_id>/<stage>/logs/   step logs and run records
   |- runs/<run_id>/<stage>/reports/  statistics and validation reports
   |- versions/<snapshot>/dvc.lock  data version per snapshot
   |- versions/<snapshot>/<run_id>.json
```
