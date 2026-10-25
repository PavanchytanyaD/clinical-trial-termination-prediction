# Data imports stay inside the fixtures, so the DAG tests also run in an
# Airflow environment without the pipeline's data libraries.
import pytest


@pytest.fixture(scope="session")
def snapshot_dir(tmp_path_factory):
    """A small placeholder AACT snapshot on disk."""
    from scripts import sample_snapshot
    out = tmp_path_factory.mktemp("aact")
    sample_snapshot.write_snapshot(out, "20261001", n_trials=300, seed=1)
    return out


@pytest.fixture()
def features(snapshot_dir):
    from scripts.preprocess import placeholder_features
    df, _ = placeholder_features(snapshot_dir)
    # same dtypes as after a CSV round trip (what the stages read)
    for c in ["nct_id", "overall_status", "start_date",
              "study_first_submitted_date"]:
        df[c] = df[c].astype("str")
    return df


@pytest.fixture()
def toy():
    import numpy as np
    import pandas as pd
    return pd.DataFrame({
        "nct_id": pd.Series([f"NCT{i:08d}" for i in range(200)], dtype="str"),
        "overall_status": pd.Series(["COMPLETED", "TERMINATED"] * 100,
                                    dtype="str"),
        "label": [0.0, 1.0] * 100,
        "start_date": pd.Series(["2020-01-01", "2021-05"] * 100, dtype="str"),
        "study_first_submitted_date": pd.Series(["2019-12-01"] * 200,
                                                dtype="str"),
        "has_dmc": np.tile([0, 1], 100).astype(np.int8),
        "n_facilities": np.arange(200, dtype=float),
        "n_other_outcomes": np.tile([0, 1], 100),
    })
