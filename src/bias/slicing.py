"""Turn raw trial columns into the label and slice columns.

Each slice function takes the trials DataFrame and returns one group name
per trial. Missing values always go to the group "UNKNOWN", so every trial
lands in a group. Therapeutic area is the one slice where a trial can be in
several groups at once; those are kept as one text value joined by "|".
"""

import numpy as np
import pandas as pd

UNKNOWN = "UNKNOWN"
STOPPED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}

# AACT booleans are True/False from the database and "t"/"f" in flat files.
_BOOL = {True: True, False: False, "t": True, "f": False,
         "True": True, "False": False}


def make_label(status):
    """1 = stopped early, 0 = completed, NaN = not finished yet (censored)."""
    label = pd.Series(np.nan, index=status.index)
    label[status.isin(STOPPED_STATUSES)] = 1.0
    label[status == "COMPLETED"] = 0.0
    return label


def as_group(values):
    """Use the raw value as the group name; missing values become UNKNOWN."""
    return values.fillna(UNKNOWN).astype(str).replace("", UNKNOWN)


def phase_slice(trials):
    return as_group(trials.phase)


def study_type_slice(trials):
    return as_group(trials.study_type)


def sponsor_class_slice(trials):
    return as_group(trials.sponsor_class)


def gender_slice(trials):
    return as_group(trials.gender)


def therapeutic_area_slice(trials):
    return as_group(trials.therapeutic_areas)


def to_bool(values):
    """True/False per row; missing or unexpected values count as False."""
    return values.map(lambda v: _BOOL.get(v, False)).astype(bool)


def age_group_slice(trials):
    """CHILD_ONLY, ADULT_ONLY (adults and/or older adults) or CHILD_AND_ADULT.
    Trials with none of the three age flags set are UNKNOWN."""
    child = to_bool(trials.child)
    adult = to_bool(trials.adult) | to_bool(trials.older_adult)
    group = pd.Series(UNKNOWN, index=trials.index)
    group[child & ~adult] = "CHILD_ONLY"
    group[~child & adult] = "ADULT_ONLY"
    group[child & adult] = "CHILD_AND_ADULT"
    return group


def region_slice(trials):
    """US_ONLY, NON_US_ONLY or US_AND_NON_US, from the trial's countries."""
    def region(countries):
        if not isinstance(countries, str) or countries == "":
            return UNKNOWN
        names = countries.split("|")
        has_us = "United States" in names
        has_other = any(name != "United States" for name in names)
        if has_us and has_other:
            return "US_AND_NON_US"
        return "US_ONLY" if has_us else "NON_US_ONLY"
    return trials.countries.map(region)


def start_era_slice(trials, edges):
    """Group by start year, e.g. edges [2000, 2010] -> before 2000,
    2000-2009, 2010 and later."""
    year = pd.to_datetime(trials.start_date, errors="coerce").dt.year
    bins = [-np.inf] + list(edges) + [np.inf]
    labels = [f"before {edges[0]}"]
    labels += [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])]
    labels += [f"{edges[-1]} and later"]
    era = pd.cut(year, bins=bins, labels=labels, right=False)
    return era.cat.add_categories(UNKNOWN).fillna(UNKNOWN).astype(str)


SLICE_FUNCTIONS = {
    "phase": phase_slice,
    "study_type": study_type_slice,
    "sponsor_class": sponsor_class_slice,
    "region": region_slice,
    "therapeutic_area": therapeutic_area_slice,
    "gender": gender_slice,
    "age_group": age_group_slice,
}


def add_slices(trials, slice_config):
    """Return nct_id, overall_status, label and one column per slice."""
    out = trials[["nct_id", "overall_status"]].copy()
    out["label"] = make_label(trials.overall_status)
    for name in slice_config["names"]:
        if name == "start_era":
            out[name] = start_era_slice(trials,
                                        slice_config["start_year_edges"])
        elif name in SLICE_FUNCTIONS:
            out[name] = SLICE_FUNCTIONS[name](trials)
        else:
            raise ValueError(f"unknown slice: {name!r}")
    return out
