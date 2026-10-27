import json

import numpy as np
import pandas as pd
import pytest

from scripts import sample_snapshot
from scripts.acquire import check_tables
from scripts.config import REQUIRED_TABLES
from scripts.preprocess import (ID_COLUMNS, build, check_contract,
                                label_from_status)
from src.error_handling import ConfigError


def test_sample_snapshot_has_every_table(snapshot_dir):
    sizes = check_tables(snapshot_dir)
    assert set(sizes) == set(REQUIRED_TABLES)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    assert manifest["source"] == "sample"
    assert manifest["snapshot"] == "20261001"


def test_sample_snapshot_is_deterministic():
    a = sample_snapshot.generate_tables(50, seed=3)
    b = sample_snapshot.generate_tables(50, seed=3)
    assert a["studies"] == b["studies"]
    c = sample_snapshot.generate_tables(50, seed=4)
    assert a["studies"] != c["studies"]


def test_sample_trial_ids_are_unique_and_valid():
    studies = sample_snapshot.generate_tables(500)["studies"]
    ids = [r["nct_id"] for r in studies]
    assert len(set(ids)) == 500
    assert all(i.startswith("NCT") and len(i) == 11 for i in ids)


def test_check_tables_rejects_a_table_without_nct_id(tmp_path, snapshot_dir):
    for t in REQUIRED_TABLES:
        (tmp_path / f"{t}.txt").write_text(
            (snapshot_dir / f"{t}.txt").read_text())
    (tmp_path / "designs.txt").write_text("id|allocation\n1|NA\n")
    with pytest.raises(ValueError, match="nct_id"):
        check_tables(tmp_path)


def test_check_tables_rejects_a_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_tables(tmp_path)


def test_label_from_status():
    labels = label_from_status(pd.Series(["TERMINATED", "WITHDRAWN",
                                          "SUSPENDED", "COMPLETED",
                                          "RECRUITING"]))
    assert labels.iloc[:4].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert np.isnan(labels.iloc[4])


def test_placeholder_features_follow_the_contract(snapshot_dir):
    df, meta, builder = build("placeholder", snapshot_dir)
    assert builder == "placeholder"
    check_contract(df)
    assert list(df.columns[:len(ID_COLUMNS)]) == ID_COLUMNS
    features = df.drop(columns=ID_COLUMNS)
    assert features.notna().all().all()  # missing values are flagged + filled
    assert meta["n_rows"] == len(df) == 300
    assert sum(meta["label_counts"].values()) == 300


def test_check_contract_rejects_duplicates(features):
    with pytest.raises(ValueError, match="duplicate"):
        check_contract(pd.concat([features, features.head(1)]))


def test_unknown_mode_is_an_error(snapshot_dir):
    with pytest.raises(ConfigError):
        build("magic", snapshot_dir)


def test_db_source_passes_the_trial_window(tmp_path, monkeypatch):
    from scripts import acquire
    seen = {}

    def fake_export(out_dir, **window):
        seen.update(window)
        sample_snapshot.write_snapshot(out_dir, "20261101", n_trials=30)

    monkeypatch.setattr(acquire.fetch_aact, "export_from_db", fake_export)
    manifest = acquire.acquire(
        {"source": "db", "snapshot": "20261101",
         "registered_from": "2019-01-01", "registered_to": "2019-12-31",
         "study_types": ["INTERVENTIONAL"]}, tmp_path / "snap")
    assert seen == {"registered_from": "2019-01-01",
                    "registered_to": "2019-12-31",
                    "study_types": ["INTERVENTIONAL"]}
    assert manifest["trials"] == seen
