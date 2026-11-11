"""The model pipeline as a Vertex AI Pipeline (Kubeflow Pipelines v2).

Every stage runs `python -m model_pipeline.run --stage <name>` in the
model-pipeline image, with the same gs:// work folder, so the graph in the
console shows each stage as its own box:

    load_data -> train -> validate -> bias -> select -> test -> register
                                                          \\-> sensitivity

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
TEMPLATE = Path(__file__).with_name("ctt_model_pipeline.json")


@dsl.container_component
def stage(name: str, workdir: str, snapshot: str, run_id: str,
          git_sha: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c",
                 'GIT_SHA="$4" python -m model_pipeline.run --stage "$0" '
                 '--workdir "$1" --snapshot "$2" --run-id "$3"'],
        args=[name, workdir, snapshot, run_id, git_sha])


@dsl.pipeline(name="ctt-model-pipeline",
              description="Train, validate, bias-check, select and register "
                          "the trial termination model")
def ctt_model_pipeline(workdir: str, snapshot: str = "latest",
                       run_id: str = "manual", git_sha: str = "manual"):
    def step(name, after=None, cpu="2", memory="8G"):
        task = stage(name=name, workdir=workdir, snapshot=snapshot,
                     run_id=run_id, git_sha=git_sha)
        task.set_display_name(name).set_caching_options(False)
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
