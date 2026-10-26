import zipfile

import pytest
import requests

from src.error_handling import DataUnavailableError
from src.ingestion import fetch_aact
from src.ingestion.fetch_aact import (
    download, export_from_db, extract_tables, snapshot_name, trial_filter,
    validate,
)

TABLES = ["studies", "eligibilities"]


def write_zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            z.writestr(name, "id|nct_id|value\n1|NCT1|x\n")


def test_extract_tables_from_subfolder(tmp_path):
    zip_path = tmp_path / "20261001_export_ctgov.zip"
    write_zip(zip_path, ["export/studies.txt", "export/eligibilities.txt",
                         "export/facilities.txt"])
    out = tmp_path / "out"
    extract_tables(zip_path, out, TABLES)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["eligibilities.txt", "studies.txt"]
    assert set(validate(out, TABLES)) == set(TABLES)


def test_extract_tables_missing_table(tmp_path):
    zip_path = tmp_path / "export.zip"
    write_zip(zip_path, ["studies.txt"])
    with pytest.raises(ValueError, match="eligibilities"):
        extract_tables(zip_path, tmp_path / "out", TABLES)


def test_extract_tables_bad_zip(tmp_path):
    bad = tmp_path / "export.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="cannot open"):
        extract_tables(bad, tmp_path / "out", TABLES)


def test_validate_rejects_bad_files(tmp_path):
    (tmp_path / "studies.txt").write_text("id|nct_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        validate(tmp_path, ["studies"])
    (tmp_path / "studies.txt").write_text("id|name\n1|x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nct_id"):
        validate(tmp_path, ["studies"])
    with pytest.raises(FileNotFoundError):
        validate(tmp_path, ["eligibilities"])


def test_snapshot_name():
    assert snapshot_name("dl/20261001_export_ctgov.zip") == "20261001"
    assert len(snapshot_name("export.zip")) == 8


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_bytes):
        yield self.body


def test_download_retries_then_succeeds(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, stream, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise requests.ConnectionError("network down")
        return FakeResponse(b"zip bytes")

    monkeypatch.setattr(fetch_aact.requests, "get", fake_get)
    dest = download("https://example.org/a.zip", tmp_path / "a.zip",
                    retries=3, backoff_seconds=0)
    assert dest.read_bytes() == b"zip bytes" and len(calls) == 3
    assert not (tmp_path / "a.zip.part").exists()


def test_download_gives_up(tmp_path, monkeypatch):
    def fake_get(url, stream, timeout):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(fetch_aact.requests, "get", fake_get)
    with pytest.raises(DataUnavailableError, match="after 2 attempts"):
        download("https://example.org/a.zip", tmp_path / "a.zip",
                 retries=2, backoff_seconds=0)
    assert not (tmp_path / "a.zip").exists()


def test_db_export_needs_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("AACT_USER", raising=False)
    monkeypatch.delenv("AACT_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="AACT_USER"):
        export_from_db(tmp_path)


def test_trial_filter_builds_one_condition_per_bound():
    where, args = trial_filter("2019-01-01", "2019-12-31", ["interventional"])
    assert where == ("study_first_submitted_date >= %s AND "
                     "study_first_submitted_date <= %s AND "
                     "study_type = ANY(%s)")
    assert args == ["2019-01-01", "2019-12-31", ["INTERVENTIONAL"]]


def test_no_trial_filter_means_every_trial():
    assert trial_filter() == (None, [])
    assert trial_filter(registered_from="2017-01-01") == (
        "study_first_submitted_date >= %s", ["2017-01-01"])
