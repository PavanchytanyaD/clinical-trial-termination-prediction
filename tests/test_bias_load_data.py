import pandas as pd
import pytest

from src.bias.load_data import (REQUIRED_COLUMNS, load_trials, query_aact,
                                read_file)


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


def test_read_file_keeps_phase_na(tmp_path):
    # AACT uses the literal phase "NA" (not applicable); it is not missing
    path = tmp_path / "trials.csv"
    path.write_text("nct_id,phase\nNCT00000001,NA\nNCT00000002,\n")
    df = read_file(path, {"nct_id": "nct_id", "phase": "phase"})
    assert df.phase.iloc[0] == "NA"
    assert pd.isna(df.phase.iloc[1])


SNAPSHOT = {
    "studies": ["nct_id|overall_status|phase|study_type|start_date",
                "NCT00000001|COMPLETED|NA|INTERVENTIONAL|2015-03-01",
                "NCT00000002|TERMINATED|PHASE2|INTERVENTIONAL|2019-01"],
    "sponsors": ["nct_id|agency_class|lead_or_collaborator",
                 "NCT00000001|INDUSTRY|lead", "NCT00000001|NIH|collaborator",
                 "NCT00000002|OTHER|lead"],
    "eligibilities": ["nct_id|gender|child|adult|older_adult",
                      "NCT00000001|ALL|f|t|t", "NCT00000002|FEMALE|f|t|f"],
    "countries": ["nct_id|name", "NCT00000001|United States",
                  "NCT00000001|Canada"],
    "browse_conditions": ["nct_id|downcase_mesh_term",
                          "NCT00000002|neoplasms",
                          "NCT00000002|carcinoma"],
}


def test_snapshot_source(tmp_path):
    for table, lines in SNAPSHOT.items():
        (tmp_path / f"{table}.txt").write_text("\n".join(lines) + "\n")
    df = load_trials({"source": "snapshot", "snapshot_dir": str(tmp_path)})
    df = df.set_index("nct_id")
    assert set(REQUIRED_COLUMNS) <= set(df.columns) | {"nct_id"}
    assert df.loc["NCT00000001", "phase"] == "NA"
    assert df.loc["NCT00000001", "sponsor_class"] == "INDUSTRY"
    assert df.loc["NCT00000001", "countries"] == "Canada|United States"
    assert pd.isna(df.loc["NCT00000002", "countries"])
    assert df.loc["NCT00000002", "therapeutic_areas"] == "neoplasms"


def test_snapshot_source_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trials({"source": "snapshot",
                     "snapshot_dir": str(tmp_path / "nope")})
