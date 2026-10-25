"""Logging setup shared by the pipeline scripts."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(name, log_dir="logs", level=logging.INFO):
    """Log to the console and to <log_dir>/<name>.log."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers,
                        force=True)


def run_main(main, name):
    """Run a script's main() with logging set up; log any error and exit 1."""
    setup_logging(name)
    try:
        main()
    except Exception:
        logging.getLogger(name).exception("%s failed", name)
        raise SystemExit(1)
