"""Sign-in to the team's MLflow server on Cloud Run.

The server (infra/mlflow/) is not public: Cloud Run lets a request in only
with a Google ID token of an account that has roles/run.invoker. MLflow
sends MLFLOW_TRACKING_TOKEN as "Authorization: Bearer <token>" on every
request, so when MLFLOW_TRACKING_URI is a *.run.app address this module puts
a fresh ID token there (they last one hour; a new one is fetched when the
current one has less than five minutes left).

The token comes from the service account the code runs as (pipeline-sa on
Vertex AI, cicd-sa in CI), or, on a laptop signed in with
`gcloud auth login`, from `gcloud auth print-identity-token`.
"""

import base64
import json
import logging
import os
import subprocess
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
TOKEN_ENV = "MLFLOW_TRACKING_TOKEN"
REFRESH_BEFORE = 300          # seconds before expiry
_cache = {"token": None, "expiry": 0.0, "audience": None}


def is_cloud_run(uri):
    u = urlparse(uri or "")
    return u.scheme == "https" and (u.hostname or "").endswith(".run.app")


def audience(uri):
    u = urlparse(uri)
    return f"{u.scheme}://{u.hostname}"


def expiry(token):
    """exp claim of a JWT (not verified: only used to know when to
    refresh); 0 if it cannot be read."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return 0.0


def iam_id_token(aud):
    """ID token from the IAM Credentials API, for the service account of
    the default credentials. Vertex AI training containers run as a service
    account but their metadata server does not issue ID tokens; the account
    needs roles/iam.serviceAccountTokenCreator on itself."""
    import google.auth
    import google.auth.transport.requests
    import requests

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    email = getattr(creds, "service_account_email", None)
    if not email or email == "default":
        email = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/"
            "service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}, timeout=10).text.strip()
    r = requests.post(
        "https://iamcredentials.googleapis.com/v1/projects/-/"
        f"serviceAccounts/{email}:generateIdToken",
        headers={"Authorization": f"Bearer {creds.token}"},
        json={"audience": aud, "includeEmail": True}, timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def fetch_id_token(aud):
    """ID token for the audience: from the service account credentials or
    the metadata server, else from the IAM Credentials API (Vertex AI), else
    from gcloud for a signed-in user."""
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token
        return google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), aud)
    except Exception as exc:
        logger.info("google.auth gave no ID token (%s); trying the IAM "
                    "Credentials API", str(exc)[:200])
    try:
        return iam_id_token(aud)
    except Exception as exc:
        # user credentials (gcloud auth application-default login) cannot
        # mint ID tokens this way; gcloud can
        logger.info("IAM Credentials API gave no ID token (%s); trying "
                    "gcloud", str(exc)[:200])
    out = subprocess.run(["gcloud", "auth", "print-identity-token"],
                         capture_output=True, text=True, timeout=60,
                         check=True)
    return out.stdout.strip()


def ensure_token(uri=None, now=None):
    """Set MLFLOW_TRACKING_TOKEN for a Cloud Run tracking URI. Returns True
    if a token of ours is in place. A token set by hand (not by this module)
    is left alone."""
    uri = uri if uri is not None else os.environ.get("MLFLOW_TRACKING_URI")
    if not is_cloud_run(uri):
        return False
    current = os.environ.get(TOKEN_ENV)
    if current and current != _cache["token"]:
        return False          # set by the user
    now = time.time() if now is None else now
    aud = audience(uri)
    if (current and _cache["audience"] == aud
            and _cache["expiry"] - now > REFRESH_BEFORE):
        return True
    try:
        token = fetch_id_token(aud)
    except Exception as exc:
        logger.warning("no ID token for %s: %s", aud, exc)
        return False
    _cache.update(token=token, expiry=expiry(token) or now + 3000,
                  audience=aud)
    os.environ[TOKEN_ENV] = token
    return True
