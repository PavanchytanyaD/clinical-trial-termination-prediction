"""Email alerts, sent with the data pipeline's alert code
(Data-Pipeline/scripts/alerts.py), so the same subscribers get both.

From a shell (CI uses this when the pipeline job fails; the SMTP_* and
ALERT_* settings come from the environment):
    python -m model_pipeline.notify "subject" --body "text"
"""

import argparse
import logging
import os
import tempfile

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
    logger.info("alert: [ctt model pipeline] %s", subject)
    try:
        _load_secret_env(ctx.params["alerts"]["secret"],
                         ctx.params["registry"]["project"])
        from scripts import alerts
        # the data pipeline's code puts its own name in front of the subject
        alerts.SUBJECT_PREFIX = "[ctt model pipeline]"
        send_alert = alerts.send_alert
        run_id = ctx.state.get("run_id", "?")
        send_alert(subject, f"Run: {run_id}\n\n{body}")
    except Exception as exc:
        logger.warning("alert not sent: %s", exc)


def main(argv=None):
    from model_pipeline.config import Context, load_params

    parser = argparse.ArgumentParser(description="send one alert email")
    parser.add_argument("subject")
    parser.add_argument("--body", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    ctx = Context(tempfile.mkdtemp(prefix="ctt-alert-"), load_params())
    alert(ctx, args.subject, args.body)


if __name__ == "__main__":
    main()
