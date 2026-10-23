import pandas as pd
import pytest

from src.bias.load_data import REQUIRED_COLUMNS, load_trials, query_aact


def test_file_source_renames_columns(tmp_path):
    # The file uses "status" and "lead_sponsor" instead of our names.
    row = {c: ["x"] for c in REQUIRED_COLUMNS}
    row["nct_id"] = ["NCT1"]
    df = pd.DataFrame(row).rename(columns={"overall_status": "status",
                                           "sponsor_class": "lead_sponsor"})
    path = tmp_path / "trials.csv"
    df.to_csv(path, index=False)

    trials = load_trials({
        "source": "file", "file_path": str(path),
        "columns": {"overall_status": "status",
                    "sponsor_class": "lead_sponsor"}})
    assert "overall_status" in trials.columns
    assert "sponsor_class" in trials.columns


def test_file_source_missing_column(tmp_path):
    path = tmp_path / "trials.csv"
    pd.DataFrame({"nct_id": ["NCT1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_trials({"source": "file", "file_path": str(path)})


def test_file_source_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trials({"source": "file",
                     "file_path": str(tmp_path / "nope.csv")})


def test_duplicate_trials_dropped(tmp_path):
    row = {c: ["x", "x"] for c in REQUIRED_COLUMNS}
    row["nct_id"] = ["NCT1", "NCT1"]
    path = tmp_path / "trials.csv"
    pd.DataFrame(row).to_csv(path, index=False)
    trials = load_trials({"source": "file", "file_path": str(path)})
    assert len(trials) == 1


def test_unknown_source():
    with pytest.raises(ValueError, match="unknown input source"):
        load_trials({"source": "bigquery"})


def test_aact_needs_credentials(monkeypatch):
    monkeypatch.delenv("AACT_USER", raising=False)
    monkeypatch.delenv("AACT_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="AACT_USER"):
        query_aact()
