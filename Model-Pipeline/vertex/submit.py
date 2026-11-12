"""Compile the pipeline and start a run on Vertex AI.

    python vertex/submit.py                       # latest data version
    python vertex/submit.py --snapshot 20261027 --wait
    python vertex/submit.py --mlflow-uri https://ctt-mlflow-....run.app

Runs as pipeline-sa; the work folder of each run is
gs://ctt-mlops-project-data/model-pipeline/runs/<run_id>/ (reports, charts,
MLflow runs unless --mlflow-uri or MLFLOW_TRACKING_URI names the team's
tracking server, the model artifact).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import compile_pipeline  # noqa: E402

PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="latest")
    p.add_argument("--git-sha", default="manual")
    p.add_argument("--mlflow-uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI", ""),
                   help="MLflow tracking server (default: "
                        "MLFLOW_TRACKING_URI, else runs in the work folder)")
    p.add_argument("--wait", action="store_true",
                   help="wait for the run and fail if it fails")
    args = p.parse_args()

    from google.cloud import aiplatform
    params = yaml.safe_load(PARAMS.read_text())
    reg, bucket = params["registry"], params["data"]["bucket"]
    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    aiplatform.init(project=reg["project"], location=reg["region"])
    job = aiplatform.PipelineJob(
        display_name=f"ctt-model-{run_id}",
        template_path=str(compile_pipeline()),
        pipeline_root=f"gs://{bucket}/model-pipeline/pipeline-root",
        parameter_values={
            "workdir": f"gs://{bucket}/model-pipeline/runs/{run_id}",
            "snapshot": args.snapshot, "run_id": run_id,
            "git_sha": args.git_sha, "mlflow_uri": args.mlflow_uri},
        enable_caching=False)
    job.submit(service_account=f"pipeline-sa@{reg['project']}"
                               ".iam.gserviceaccount.com")
    print(f"submitted {run_id}: {job.resource_name}")
    print(f"MLflow: {args.mlflow_uri or 'runs in the work folder'}")
    print("follow it in the console: Vertex AI (Agent Platform) > Pipelines")
    if args.wait:
        job.wait()


if __name__ == "__main__":
    main()
