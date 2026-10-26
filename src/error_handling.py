"""Shared error handling for pipeline steps.

Builds on ``logging_utils.setup_logging`` (the shared logger): that module
decides *where* logs go (console + ``logs/<name>.log``); this module decides
*how each pipeline step reports success and failure*.

Typical use inside any step module (ingestion, features, model, agent):

    from src.logging_utils import setup_logging
    from src.error_handling import log_stage, DataUnavailableError, require

    @log_stage("fetch_aact")
    def main():
        require(snapshot.exists(), f"snapshot missing: {snapshot}",
                DataUnavailableError)
        ...

Or wrap a single stage inside a larger function:

    with stage("clean text"):
        df = clean(df)
"""

import functools
import logging
import time
from contextlib import contextmanager


class PipelineError(Exception):
    """Base class for all expected pipeline failures."""


# Each error also subclasses the built-in exception it replaces, so code and
# tests that catch FileNotFoundError / ValueError / RuntimeError keep working.
class DataUnavailableError(PipelineError, FileNotFoundError):
    """A source (AACT snapshot, export file, DB) is missing, empty, or unreachable."""


class DataValidationError(PipelineError, ValueError):
    """Data failed a schema, column, or value-range check."""


class ConfigError(PipelineError, RuntimeError):
    """A required setting or environment variable is missing or invalid."""


def _logger(name=None):
    return logging.getLogger(name if name else "pipeline")


def log_stage(stage_name, logger_name=None):
    """Decorator: log START / DONE / FAILED for a function, with timing.

    On failure it logs one FAILED line with the error and re-raises, so an
    orchestrator (Airflow, ``run_main`` or ``tracked_step``) still sees the error
    and the step exits non-zero. The full traceback is logged once, by that outer
    handler, not here as well. BaseException is caught so that a sys.exit()
    inside the stage is reported too.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = _logger(logger_name or stage_name)
            log.info("START %s", stage_name)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except BaseException as e:
                log.error("FAILED %s after %.1fs: %r", stage_name, time.perf_counter() - t0, e)
                raise
            log.info("DONE  %s in %.1fs", stage_name, time.perf_counter() - t0)
            return result

        return wrapper

    return decorator


@contextmanager
def stage(stage_name, logger_name=None):
    """Context-manager form of ``log_stage`` for part of a function."""
    log = _logger(logger_name or stage_name)
    log.info("START %s", stage_name)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as e:
        log.error("FAILED %s after %.1fs: %r", stage_name, time.perf_counter() - t0, e)
        raise
    log.info("DONE  %s in %.1fs", stage_name, time.perf_counter() - t0)


def require(condition, message, error=DataValidationError):
    """Raise ``error(message)`` if ``condition`` is falsy. A readable assert.

    require(path.exists(), f"snapshot missing: {path}", DataUnavailableError)
    require("nct_id" in df.columns, "nct_id column missing")
    """
    if not condition:
        raise error(message)
