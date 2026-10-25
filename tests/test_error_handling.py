"""Unit tests for src/error_handling.py."""

import logging

import pytest

from src.error_handling import (
    DataUnavailableError,
    DataValidationError,
    PipelineError,
    log_stage,
    require,
    stage,
)


def test_log_stage_success(caplog):
    @log_stage("demo_step")
    def work(x):
        return x * 2

    with caplog.at_level(logging.INFO):
        assert work(3) == 6
    msgs = " ".join(r.message for r in caplog.records)
    assert "START demo_step" in msgs and "DONE  demo_step" in msgs


def test_log_stage_reraises(caplog):
    @log_stage("boom_step")
    def work():
        raise ValueError("nope")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            work()
    assert any("FAILED boom_step" in r.message for r in caplog.records)


def test_stage_contextmanager(caplog):
    with caplog.at_level(logging.INFO):
        with stage("inline"):
            pass
    msgs = " ".join(r.message for r in caplog.records)
    assert "START inline" in msgs and "DONE  inline" in msgs


def test_require_passes():
    require(True, "should not raise")


def test_require_raises_default():
    with pytest.raises(DataValidationError):
        require(False, "bad")


def test_require_custom_error():
    with pytest.raises(DataUnavailableError):
        require(False, "missing", DataUnavailableError)


def test_exception_hierarchy():
    assert issubclass(DataUnavailableError, PipelineError)
    assert issubclass(DataValidationError, PipelineError)
