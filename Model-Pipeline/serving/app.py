"""Prediction server for the trial termination model, in the format of a
Vertex AI custom serving container.

Vertex AI starts the container with these settings (defaults for a local
run in brackets):

    AIP_HTTP_PORT      port to listen on [8080]
    AIP_HEALTH_ROUTE   health check route [/health]
    AIP_PREDICT_ROUTE  prediction route [/predict]
    AIP_STORAGE_URI    gs:// folder with the model artifact (a copy of the
                       registry version's artifact_uri); a local folder works
                       too. MODEL_DIR is used when it is not set.

The artifact is the test stage's artifact/ folder (model.joblib,
model_card.json, calibrator.joblib and, for a text model, text_pca.joblib;
see model_pipeline/artifact.py). Request and response:

    POST /predict
    {"instances": [{"f1": 0.3, "f2": 1, ..., "region": "US_only",
                    "phase": "PHASE2"}, ...]}
    -> {"predictions": [{"probability": 0.71, "score": 0.66, "label": 1,
                         "threshold": 0.5, "calibrated_probability": 0.31,
                         "groups": {"region": "US_only", "phase": "PHASE2"}},
                        ...]}

An instance is either an object keyed by feature name (extra keys are
ignored; the model's threshold slices, e.g. region and phase, pick the
group thresholds) or a list of values in the model's feature order (then
parameters.groups may give the groups). A text model also takes
"text_embeddings": {"title": [768 numbers], ...} (the data pipeline's
BioBERT vectors) instead of its txt_* features. probability is the model's
own probability, score the score after the group thresholds (compared with
threshold), label 1 means "will stop early", calibrated_probability the
probability of stopping to show a reviewer.

Local run:
    MODEL_DIR=runs/local/artifact python -m serving.app
"""

import logging
import math
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model_pipeline.artifact import load

logger = logging.getLogger("serving")


class PredictRequest(BaseModel):
    instances: list[Union[dict[str, Any], list[Any]]]
    parameters: Optional[dict[str, Any]] = None


def download(uri, dest):
    """Copy a gs:// folder (all objects under the prefix) to dest."""
    from google.cloud import storage

    bucket_name, _, prefix = uri.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    client = storage.Client()
    count = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        target = Path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        count += 1
    if count == 0:
        raise FileNotFoundError(f"nothing under {uri}")
    return Path(dest)


def model_folder():
    uri = os.environ.get("AIP_STORAGE_URI") or os.environ.get("MODEL_DIR")
    if not uri:
        raise RuntimeError("set AIP_STORAGE_URI or MODEL_DIR")
    if uri.startswith("gs://"):
        return download(uri, tempfile.mkdtemp(prefix="model-"))
    return Path(uri)


def _number(value, column):
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{column}: not a number ({value!r})")


def to_rows(model, instances, parameters):
    """Feature matrix (list of lists) and the groups of each instance
    ({slice: [group per instance]} for the model's threshold slices)."""
    rows = []
    groups = {name: [] for name in model.group_slices}
    given = (parameters or {}).get("groups")
    for i, inst in enumerate(instances):
        if isinstance(inst, dict):
            if "text_embeddings" in inst:
                try:
                    inst = {**inst, **model.text_values(
                        inst["text_embeddings"] or {})}
                except ValueError as exc:
                    raise HTTPException(400, f"instance {i}: {exc}")
            missing = model.missing_columns(inst)
            if missing:
                raise HTTPException(
                    400, f"instance {i}: missing features "
                         f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
            rows.append([_number(inst[c], c) for c in model.columns])
            found = {name: inst.get(name) for name in model.group_slices}
        else:
            if len(inst) != len(model.columns):
                raise HTTPException(
                    400, f"instance {i}: {len(inst)} values, the model "
                         f"needs {len(model.columns)}")
            rows.append([_number(v, c) for v, c in zip(inst, model.columns)])
            g = given[i] if given and i < len(given) else None
            found = g if isinstance(g, dict) else \
                {model.group_slice: g} if model.group_slice else {}
        for name in model.group_slices:
            value = found.get(name)
            groups[name].append("UNKNOWN" if value is None else str(value))
    return rows, groups


def predict(model, instances, parameters=None):
    rows, groups = to_rows(model, instances, parameters)
    if not rows:
        return []
    try:
        prob = model.probabilities(rows)
        score = model.scores(rows, groups) if model.has_groups else prob
    except ValueError as exc:   # e.g. missing values for a model without
        raise HTTPException(400, f"cannot score: {exc}")  # NaN support
    risk = model.calibrated(score)
    out = []
    for i, (p, s) in enumerate(zip(prob, score)):
        own = {name: groups[name][i] for name in model.group_slices}
        item = {"probability": round(float(p), 6),
                "score": round(float(s), 6),
                "label": int(s >= model.threshold),
                "threshold": round(model.group_threshold(
                    own.get(model.group_slice)), 6)}
        if risk is not None:
            item["calibrated_probability"] = round(float(risk[i]), 6)
        if len(own) == 1:
            item["group"] = next(iter(own.values()))
        elif own:
            item["groups"] = own
        out.append(item)
    return out


def create_app(model_dir=None):
    """The FastAPI app; model_dir loads that folder now (tests), otherwise
    the model is loaded at startup from AIP_STORAGE_URI or MODEL_DIR."""
    state = {"model": None, "error": None}

    def _load(folder):
        try:
            state["model"] = load(folder)
            logger.info("loaded %s (%d features) from %s",
                        state["model"].name, len(state["model"].columns),
                        folder)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("model not loaded")

    @asynccontextmanager
    async def lifespan(app):
        if state["model"] is None and state["error"] is None:
            try:
                _load(model_folder())
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("model not loaded")
        yield

    if model_dir is not None:
        _load(model_dir)

    app = FastAPI(title="ctt-termination-model", lifespan=lifespan)
    health_route = os.environ.get("AIP_HEALTH_ROUTE", "/health")
    predict_route = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

    def health():
        model = state["model"]
        if model is None:
            return JSONResponse({"status": "not ready",
                                 "error": state["error"]}, status_code=503)
        return {"status": "ok", "model": model.name,
                "n_features": len(model.columns),
                "threshold": model.threshold,
                "group_slices": model.group_slices,
                "calibrated": model.calibrator is not None,
                "text_features": model.text is not None}

    def predict_route_fn(body: PredictRequest):
        if state["model"] is None:
            raise HTTPException(503, f"model not loaded: {state['error']}")
        return {"predictions": predict(state["model"], body.instances,
                                       body.parameters)}

    app.add_api_route(health_route, health, methods=["GET"])
    app.add_api_route(predict_route, predict_route_fn, methods=["POST"])
    app.state.serving = state
    return app


app = create_app()


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: "
                               "%(message)s")
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("AIP_HTTP_PORT", "8080")))


if __name__ == "__main__":
    main()
