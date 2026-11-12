"""The prediction server (serving/app.py) with a tiny fitted model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from model_pipeline import artifact
from serving import app as serving
from tests.helpers import (COLUMNS, CUTS, Constant, fit_logreg, make_rows,
                           write_artifact)


@pytest.fixture(scope="module")
def rows():
    return make_rows()


@pytest.fixture(scope="module")
def model(rows):
    return fit_logreg(rows)


def client_for(folder):
    return TestClient(serving.create_app(folder))


def test_health_and_predict_plain_model(tmp_path, rows, model):
    client = client_for(write_artifact(tmp_path / "a", model))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["n_features"] == 3

    sample = rows.head(5)
    body = {"instances": sample[COLUMNS + ["region"]].to_dict("records")}
    out = client.post("/predict", json=body).json()["predictions"]
    expected = model.predict_proba(sample[COLUMNS].to_numpy())[:, 1]
    assert [p["probability"] for p in out] == \
        pytest.approx(expected, abs=1e-6)
    assert [p["label"] for p in out] == [int(e >= 0.5) for e in expected]
    assert all(p["threshold"] == 0.5 for p in out)


def test_group_thresholds_change_the_label(tmp_path, model):
    folder = write_artifact(tmp_path / "a", model, "logreg+group_thresholds",
                            cuts=CUTS)
    client = client_for(folder)
    # the same trial in three regions; pick features with probability ~0.5
    base = {"f0": 0.2, "f1": 0.0, "f2": 0.0}
    p = model.predict_proba(np.array([[0.2, 0.0, 0.0]]))[0, 1]
    assert 0.3 < p < 0.7
    body = {"instances": [{**base, "region": r}
                          for r in ("US_only", "non_US", "mixed")]}
    out = client.post("/predict", json=body).json()["predictions"]
    assert [o["label"] for o in out] == [1, 0, int(p >= 0.5)]
    assert [o["threshold"] for o in out] == [0.3, 0.7, 0.5]
    assert all(o["probability"] == pytest.approx(p, abs=1e-6) for o in out)
    assert out[0]["group"] == "US_only"


def test_serving_scores_match_the_pipeline(tmp_path, rows, model):
    """Same scores as the pipeline's score_candidate (bias_model's shift)."""
    from model_pipeline.models import score_candidate

    folder = write_artifact(tmp_path / "a", model, "logreg+group_thresholds",
                            cuts=CUTS)
    cand = {"cuts": CUTS, "threshold_slice": "region"}
    pipeline = score_candidate(cand, model, rows, COLUMNS, 0.5)
    packaged = artifact.load(folder).score_frame(rows)
    assert packaged == pytest.approx(pipeline, abs=1e-9)


def test_list_instances_and_groups_parameter(tmp_path, model):
    folder = write_artifact(tmp_path / "a", model, "x+group_thresholds",
                            cuts=CUTS)
    client = client_for(folder)
    body = {"instances": [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
            "parameters": {"groups": ["US_only", "non_US"]}}
    out = client.post("/predict", json=body).json()["predictions"]
    assert [o["label"] for o in out] == [1, 0]


def test_bad_requests(tmp_path, model):
    client = client_for(write_artifact(tmp_path / "a", model))
    missing = client.post("/predict", json={"instances": [{"f0": 1}]})
    assert missing.status_code == 400 and "f1" in missing.json()["detail"]
    short = client.post("/predict", json={"instances": [[1, 2]]})
    assert short.status_code == 400
    text = client.post("/predict", json={"instances": [
        {"f0": "a", "f1": 0, "f2": 0}]})
    assert text.status_code == 400


def test_generic_object_with_predict_proba(tmp_path):
    client = client_for(write_artifact(tmp_path / "a", Constant(0.8)))
    out = client.post("/predict", json={"instances": [[0, 0, 0]]}).json()
    assert out["predictions"][0] == {"probability": 0.8, "score": 0.8,
                                     "label": 1, "threshold": 0.5}


def test_not_ready_without_a_model(tmp_path):
    client = client_for(tmp_path / "empty")
    assert client.get("/health").status_code == 503
    assert client.post("/predict",
                       json={"instances": [[0, 0, 0]]}).status_code == 503


def test_vertex_routes_and_storage_uri(tmp_path, monkeypatch, model):
    write_artifact(tmp_path / "a", model)
    monkeypatch.setenv("AIP_HEALTH_ROUTE", "/v1/health")
    monkeypatch.setenv("AIP_PREDICT_ROUTE", "/v1/models/m:predict")
    monkeypatch.setenv("AIP_STORAGE_URI", str(tmp_path / "a"))
    with TestClient(serving.create_app()) as client:   # runs the startup
        assert client.get("/v1/health").status_code == 200
        r = client.post("/v1/models/m:predict",
                        json={"instances": [[0.1, 0.2, 0.3]]})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 1


def test_download_from_gcs(tmp_path, monkeypatch):
    """AIP_STORAGE_URI gs://...: every object under the prefix is copied."""
    import google.cloud.storage as storage

    class Blob:
        def __init__(self, name):
            self.name = name

        def download_to_filename(self, path):
            with open(path, "w") as f:
                f.write(self.name)

    class Client:
        def list_blobs(self, bucket, prefix):
            assert (bucket, prefix) == ("b", "models/run-1/")
            return [Blob("models/run-1/model.joblib"),
                    Blob("models/run-1/model_card.json")]

    monkeypatch.setattr(storage, "Client", Client)
    out = serving.download("gs://b/models/run-1", tmp_path / "m")
    assert sorted(p.name for p in out.iterdir()) == \
        ["model.joblib", "model_card.json"]
