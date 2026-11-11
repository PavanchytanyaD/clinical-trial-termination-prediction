"""Email alerts, sent with the data pipeline's alert code
(Data-Pipeline/scripts/alerts.py), so the same subscribers get both."""

import logging
import os

logger = logging.getLogger(__name__)


def _load_secret_env(secret, project):
    """Put the lines of a Secret Manager secret (KEY=value, like .env) into
    the environment, for settings that are not set yet."""
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/{secret}/versions/latest"
        text = client.access_secret_version(name=name).payload.data.decode()
    except Exception as exc:  # no secret or no access: alerts are optional
        logger.info("alert settings secret not read (%s)", exc)
        return
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def alert(ctx, subject, body):
    """Send one alert email; never stops the pipeline if sending fails."""
    subject = f"[ctt model pipeline] {subject}"
    logger.info("alert: %s", subject)
    try:
        _load_secret_env(ctx.params["alerts"]["secret"],
                         ctx.params["registry"]["project"])
        from scripts.alerts import send_alert
        run_id = ctx.state.get("run_id", "?")
        send_alert(subject, f"Run: {run_id}\n\n{body}")
    except Exception as exc:
        logger.warning("alert not sent: %s", exc)
