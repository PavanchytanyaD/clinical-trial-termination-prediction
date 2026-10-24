import pandas as pd
import pytest

from src.bias.slicing import UNKNOWN, add_slices, age_group_slice, \
    make_label, region_slice, start_era_slice


def make_trials():
    return pd.DataFrame({
        "nct_id": ["NCT1", "NCT2", "NCT3", "NCT4"],
        "overall_status": ["COMPLETED", "TERMINATED", "RECRUITING", None],
        "phase": ["PHASE2", "PHASE3", None, ""],
        "study_type": ["INTERVENTIONAL"] * 4,
        "start_date": ["2005-03-01", "2019-01-15", None, "not a date"],
        "sponsor_class": ["INDUSTRY", "NIH", None, "OTHER"],
        "gender": ["ALL", "FEMALE", None, "MALE"],
        "child": [True, "f", None, False],
        "adult": [True, "t", None, False],
        "older_adult": [False, "t", None, True],
        "countries": ["United States", "France|United States", None,
                      "France"],
        "therapeutic_areas": ["neoplasms", "infections|neoplasms", None,
                              None],
    })


def test_make_label():
    status = pd.Series(["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED",
                        "RECRUITING", None])
    label = make_label(status)
    assert label[:4].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert label[4:].isna().all()


def test_age_group_handles_bools_and_text():
    groups = age_group_slice(make_trials()).tolist()
    assert groups == ["CHILD_AND_ADULT", "ADULT_ONLY", UNKNOWN, "ADULT_ONLY"]


def test_age_group_child_only():
    trials = pd.DataFrame({"child": ["t"], "adult": ["f"],
                           "older_adult": ["f"]})
    assert age_group_slice(trials).tolist() == ["CHILD_ONLY"]


def test_region():
    assert region_slice(make_trials()).tolist() == [
        "US_ONLY", "US_AND_NON_US", UNKNOWN, "NON_US_ONLY"]


def test_start_era_bins_and_bad_dates():
    eras = start_era_slice(make_trials(), [2000, 2010, 2018]).tolist()
    assert eras == ["2000-2009", "2018 and later", UNKNOWN, UNKNOWN]


def test_start_era_edges_are_inclusive_on_the_left():
    trials = pd.DataFrame({"start_date": ["1999-12-31", "2010-01-01"]})
    assert start_era_slice(trials, [2000, 2010]).tolist() == [
        "before 2000", "2010 and later"]


def test_add_slices_every_trial_gets_one_group():
    names = ["phase", "study_type", "sponsor_class", "region",
             "therapeutic_area", "gender", "age_group", "start_era"]
    sliced = add_slices(make_trials(),
                        {"names": names, "start_year_edges": [2000, 2010]})
    assert len(sliced) == 4
    for name in names:
        assert sliced[name].notna().all()
        assert (sliced[name] != "").all()
    assert sliced.phase.tolist() == ["PHASE2", "PHASE3", UNKNOWN, UNKNOWN]


def test_add_slices_unknown_name():
    with pytest.raises(ValueError, match="unknown slice"):
        add_slices(make_trials(), {"names": ["not_a_slice"]})
