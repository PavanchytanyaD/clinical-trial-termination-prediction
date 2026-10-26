"""Email alerts over SMTP.

Settings come from environment variables, or from Data-Pipeline/.env when a
variable is not set (see .env.example):

    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD,
    SMTP_STARTTLS (true), ALERT_EMAIL_FROM, ALERT_EMAIL_TO (comma separated)

When SMTP is not configured the alert is logged and nothing is sent, so the
pipeline still runs without mail settings. Sending never raises:
a broken mail server must not hide the original pipeline error.
"""

import logging
import os
import smtplib
import socket
from email.message import EmailMessage

from scripts.config import load_env_file

logger = logging.getLogger(__name__)

SUBJECT_PREFIX = "[ctt-data-pipeline]"


def smtp_settings():
    file_values = load_env_file()

    def get(name, default=""):
        return os.environ.get(name) or file_values.get(name) or default

    recipients = [r.strip() for r in get("ALERT_EMAIL_TO").split(",")
                  if r.strip()]
    return {
        "host": get("SMTP_HOST"),
        "port": int(get("SMTP_PORT", "587")),
        "user": get("SMTP_USER"),
        "password": get("SMTP_PASSWORD"),
        "starttls": get("SMTP_STARTTLS", "true").lower() != "false",
        "sender": get("ALERT_EMAIL_FROM") or get("SMTP_USER"),
        "recipients": recipients,
    }


def send_alert(subject, body, settings=None):
    """Email an alert. Returns True if it was sent."""
    s = settings or smtp_settings()
    subject = f"{SUBJECT_PREFIX} {subject}"
    if not (s["host"] and s["sender"] and s["recipients"]):
        logger.warning("alert not emailed (SMTP not configured): %s", subject)
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = s["sender"]
    msg["To"] = ", ".join(s["recipients"])
    msg.set_content(f"{body}\n\nhost: {socket.gethostname()}")
    try:
        with smtplib.SMTP(s["host"], s["port"], timeout=30) as smtp:
            if s["starttls"]:
                smtp.starttls()
            if s["user"]:
                smtp.login(s["user"], s["password"])
            smtp.send_message(msg)
    except (OSError, smtplib.SMTPException) as e:
        logger.error("could not email alert %r: %s", subject, e)
        return False
    logger.info("emailed alert %r to %s", subject, msg["To"])
    return True
