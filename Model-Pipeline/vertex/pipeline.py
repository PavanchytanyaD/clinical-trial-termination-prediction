"""The model pipeline as a Vertex AI Pipeline (Kubeflow Pipelines v2).

Every stage runs `python -m model_pipeline.run --stage <name>` in the
model-pipeline image, with the same gs:// work folder, so the graph in the
console shows each stage as its own box:

    load_data -> train -> validate -> bias -> select -> test -> register
                                                          \\-> sensitivity

MLflow: the run parameter mlflow_uri (the team's tracking server, see
infra/mlflow/) becomes MLFLOW_TRACKING_URI in every stage; empty means a
folder of runs in the work folder. CTT_SERVING_IMAGE at compile time is
passed to the register stage (the prediction image of the same commit).

Compile only: python vertex/pipeline.py
(writes vertex/ctt_model_pipeline.json)
"""

import os
from pathlib import Path

from kfp import compiler, dsl

IMAGE = os.environ.get(
    "CTT_MODEL_IMAGE",
    "us-central1-docker.pkg.dev/ctt-mlops-project/ctt-images/"
    "model-pipeline:latest")
SERVING_IMAGE = os.environ.get("CTT_SERVING_IMAGE", "")
TEMPLATE = Path(__file__).with_name("ctt_model_pipeline.json")


@dsl.container_component
def stage(name: str, workdir: str, snapshot: str, run_id: str,
          git_sha: str, mlflow_uri: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c",
                 'if [ -n "$5" ]; then export MLFLOW_TRACKING_URI="$5"; fi; '
                 'GIT_SHA="$4" python -m model_pipeline.run --stage "$0" '
                 '--workdir "$1" --snapshot "$2" --run-id "$3"'],
        args=[name, workdir, snapshot, run_id, git_sha, mlflow_uri])


@dsl.pipeline(name="ctt-model-pipeline",
              description="Train, validate, bias-check, select and register "
                          "the trial termination model")
def ctt_model_pipeline(workdir: str, snapshot: str = "latest",
                       run_id: str = "manual", git_sha: str = "manual",
                       mlflow_uri: str = ""):
    def step(name, after=None, cpu="2", memory="8G"):
        task = stage(name=name, workdir=workdir, snapshot=snapshot,
                     run_id=run_id, git_sha=git_sha, mlflow_uri=mlflow_uri)
        task.set_display_name(name).set_caching_options(False)
        if SERVING_IMAGE:
            task.set_env_variable("CTT_SERVING_IMAGE", SERVING_IMAGE)
        task.set_cpu_limit(cpu).set_memory_limit(memory)
        if after is not None:
            task.after(after)
        return task

    load = step("load_data")
    train = step("train", load, cpu="8", memory="32G")
    validate = step("validate", train)
    bias = step("bias", validate)
    select = step("select", bias)
    test = step("test", select)
    step("sensitivity", test, cpu="4", memory="16G")
    step("register", test)


def compile_pipeline(path=TEMPLATE):
    compiler.Compiler().compile(ctt_model_pipeline, str(path))
    return path


if __name__ == "__main__":
    print(compile_pipeline())
