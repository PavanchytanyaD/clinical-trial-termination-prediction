"""Run the model pipeline, one stage or all of them.

    python -m model_pipeline.run --stage all --source local --skip-register
    python -m model_pipeline.run --stage train \
        --workdir gs://BUCKET/model-pipeline/runs/RUN_ID

Stages, in order: load_data, train, validate, bias, select, calibrate,
test, lead_time, sensitivity, register. With a gs:// work folder (as on Vertex AI, where each
stage is its own container) the folder is copied down before the stage and
back up after it, so the stages hand files to each other through the bucket.
"""

import argparse
import importlib
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

from model_pipeline import notify
from model_pipeline.config import Context, load_params

STAGES = {"load_data": "data", "train": "train", "validate": "validate",
          "bias": "bias", "select": "select", "calibrate": "calibrate",
          "test": "test_set", "lead_time": "lead_time",
          "sensitivity": "sensitivity", "register": "registry"}
LOCAL_WORK = Path("/tmp/ctt-model-work")
logger = logging.getLogger("model_pipeline")


def _sync(remote, local, down):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    remote = remote.removeprefix("gs://").rstrip("/")
    if down:
        if not fs.exists(remote):
            return
        for path in fs.find(remote):
            dest = local / Path(path).relative_to(remote)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(path, str(dest))
    else:
        for f in local.rglob("*"):
            if f.is_file():
                fs.put_file(str(f), f"{remote}/{f.relative_to(local)}")


def run_stage(ctx, name):
    module = importlib.import_module(f"model_pipeline.{STAGES[name]}")
    start = time.time()
    logger.info("stage %s: start", name)
    result = module.run(ctx)
    logger.info("stage %s: done in %.0f s", name, time.time() - start)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="CTT model pipeline")
    parser.add_argument("--stage", default="all",
                        choices=["all", *STAGES])
    parser.add_argument("--workdir", default="runs/local",
                        help="local folder or gs://bucket/path")
    parser.add_argument("--snapshot", help="data version, e.g. 20261027")
    parser.add_argument("--source", choices=["lock", "local"])
    parser.add_argument("--run-id")
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: "
                               "%(message)s")
    remote = args.workdir if args.workdir.startswith("gs://") else None
    work = LOCAL_WORK if remote else Path(args.workdir)
    if remote:
        shutil.rmtree(work, ignore_errors=True)
        _sync(remote, work, down=True)
    ctx = Context(work, load_params(), args.snapshot, args.source)
    if "run_id" not in ctx.state:
        ctx.update_state(run_id=args.run_id or
                         time.strftime("run-%Y%m%d-%H%M%S"))

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    if args.skip_register and "register" in stages:
        stages.remove("register")
    try:
        for name in stages:
            run_stage(ctx, name)
        if args.stage in ("all", "register"):
            notify.alert(ctx, "training run finished",
                         (ctx.path("reports", "selection.md").read_text()
                          if ctx.path("reports", "selection.md").exists()
                          else "done"))
    except SystemExit as stop:      # a gate stopped the run on purpose
        logger.error("stopped: %s", stop)
        raise SystemExit(1)
    except Exception:
        notify.alert(ctx, f"stage {name} failed",
                     traceback.format_exc()[-3000:])
        raise
    finally:
        if remote:
            _sync(remote, work, down=False)


if __name__ == "__main__":
    main()
