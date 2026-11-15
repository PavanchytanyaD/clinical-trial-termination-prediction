"""ID tokens for the MLflow server on Cloud Run (google.auth mocked)."""

import base64
import json

import pytest

from model_pipeline import mlflow_auth

URI = "https://ctt-mlflow-abc123-uc.a.run.app"


def jwt(exp):
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(mlflow_auth.TOKEN_ENV, raising=False)
    monkeypatch.setattr(mlflow_auth, "_cache",
                        {"token": None, "expiry": 0.0, "audience": None})


@pytest.fixture
def fetches(monkeypatch):
    calls = []

    def fake(request, audience):
        calls.append(audience)
        return jwt(1000 + 3600 * len(calls))

    import google.oauth2.id_token
    monkeypatch.setattr(google.oauth2.id_token, "fetch_id_token", fake)
    return calls


def test_only_cloud_run_uris():
    assert mlflow_auth.is_cloud_run(URI)
    assert mlflow_auth.is_cloud_run(URI + "/")
    assert not mlflow_auth.is_cloud_run("http://localhost:5000")
    assert not mlflow_auth.is_cloud_run("file:///tmp/mlruns")
    assert not mlflow_auth.is_cloud_run(None)
    assert mlflow_auth.audience(URI + "/api/2.0") == URI


def test_token_set_and_reused_until_near_expiry(fetches, monkeypatch):
    import os
    assert mlflow_auth.ensure_token(URI + "/", now=1000)
    first = os.environ[mlflow_auth.TOKEN_ENV]
    assert fetches == [URI]
    assert mlflow_auth.expiry(first) == 4600
    mlflow_auth.ensure_token(URI, now=2000)          # still fresh
    assert len(fetches) == 1
    mlflow_auth.ensure_token(URI, now=4400)          # < 5 min left
    assert len(fetches) == 2
    assert os.environ[mlflow_auth.TOKEN_ENV] != first


def test_no_token_for_other_servers(fetches):
    assert not mlflow_auth.ensure_token("http://localhost:5000")
    assert fetches == []


def test_token_set_by_hand_is_kept(fetches, monkeypatch):
    import os
    monkeypatch.setenv(mlflow_auth.TOKEN_ENV, "mine")
    assert not mlflow_auth.ensure_token(URI)
    assert os.environ[mlflow_auth.TOKEN_ENV] == "mine" and fetches == []


def test_gcloud_fallback_for_user_credentials(monkeypatch):
    import subprocess

    import google.oauth2.id_token

    def no_sa(request, audience):
        raise ValueError("user credentials")

    def no_iam(aud):
        raise RuntimeError("not a service account")

    monkeypatch.setattr(google.oauth2.id_token, "fetch_id_token", no_sa)
    monkeypatch.setattr(mlflow_auth, "iam_id_token", no_iam)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": jwt(9999) + "\n"})())
    assert mlflow_auth.fetch_id_token(URI) == jwt(9999)


def test_iam_credentials_fallback_on_vertex_ai(monkeypatch):
    """Vertex AI: the metadata server gives no ID token, the IAM
    Credentials API does (for the job's service account)."""
    import google.auth
    import google.oauth2.id_token
    import requests

    def no_metadata_token(request, audience):
        raise ValueError("please provide a user-specified service account")

    class Creds:
        token = "access-token"
        service_account_email = "pipeline-sa@p.iam.gserviceaccount.com"

        def refresh(self, request):
            pass

    calls = []

    def post(url, headers, json, timeout):
        calls.append((url, headers["Authorization"], json["audience"]))
        return type("R", (), {"raise_for_status": lambda self: None,
                              "json": lambda self: {"token": jwt(7777)}})()

    monkeypatch.setattr(google.oauth2.id_token, "fetch_id_token",
                        no_metadata_token)
    monkeypatch.setattr(google.auth, "default",
                        lambda scopes=None: (Creds(), "p"))
    monkeypatch.setattr(requests, "post", post)
    assert mlflow_auth.fetch_id_token(URI) == jwt(7777)
    url, auth, aud = calls[0]
    assert "pipeline-sa@p.iam.gserviceaccount.com:generateIdToken" in url
    assert auth == "Bearer access-token"
    assert aud == mlflow_auth.audience(URI)


def test_tracking_setup_fetches_a_token(tmp_path, fetches, monkeypatch):
    """tracking._setup calls the helper before MLflow talks to the server."""
    from model_pipeline import tracking
    from model_pipeline.config import Context, load_params

    if tracking.mlflow is None:
        pytest.skip("mlflow not installed")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri",
                        lambda uri: None)
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: None)
    tracking._setup(Context(tmp_path, load_params()))
    assert fetches == [URI]
