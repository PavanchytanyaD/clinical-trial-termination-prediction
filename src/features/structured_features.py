"""Clean structured AACT fields and build design features per trial.

One row per trial (nct_id). Only fields known when a trial is registered are
used. Fields filled in after a trial ends (actual enrollment, completion
dates, results, why_stopped) are left out; see LEAKAGE_EXCLUDED. Columns in
LEAKAGE_RISK are design fields that sponsors can edit after a trial stops
(sites are often removed from withdrawn trials), so check them before use.

Categoricals are one-hot encoded against fixed category lists, so every
snapshot gets the same columns, with explicit _missing and _other levels.
Numeric fields get a _missing flag and are filled with 0, so nothing is
fitted on the full data.

    python -m src.features.structured_features --export-dir <dir>
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_text_sample import read_chunks
from src.features.text_cleaning import split_eligibility
from src.logging_utils import run_main

logger = logging.getLogger(__name__)

STOPPED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}

LEAKAGE_EXCLUDED = {
    "studies.enrollment, enrollment_type":
        "updated to actual enrollment when a trial ends (0 for withdrawn)",
    "studies.start_date_type":
        "stays ESTIMATED for trials that never start",
    "studies.primary_completion_date, completion_date and their _type":
        "updated to the actual stop date",
    "studies.why_stopped":
        "recorded stop reason; agent reference only",
    "studies.overall_status, last_known_status":
        "outcome; overall_status is used only for the label",
    "studies results_*, disposition_*, last_update_* dates":
        "written after the trial",
    "calculated_values.actual_duration, were_results_reported, "
    "months_to_report_results, number_of_sae_subjects, "
    "number_of_nsae_subjects":
        "results",
    "countries.removed":
        "set when countries are dropped from a trial",
}
LEAKAGE_RISK = ["n_facilities", "has_us_facility", "has_single_facility",
                "n_countries"]

CATEGORIES = {
    "study_type": ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"],
    "phase": ["NA", "EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2", "PHASE2",
              "PHASE2/PHASE3", "PHASE3", "PHASE4"],
    "allocation": ["RANDOMIZED", "NON_RANDOMIZED", "NA"],
    "intervention_model": ["PARALLEL", "SINGLE_GROUP", "CROSSOVER",
                           "SEQUENTIAL", "FACTORIAL"],
    "observational_model": ["COHORT", "CASE_ONLY", "CASE_CONTROL",
                            "CASE_CROSSOVER", "ECOLOGIC_OR_COMMUNITY",
                            "DEFINED_POPULATION", "FAMILY_BASED", "OTHER"],
    "primary_purpose": ["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE",
                        "BASIC_SCIENCE", "DIAGNOSTIC",
                        "HEALTH_SERVICES_RESEARCH", "SCREENING",
                        "DEVICE_FEASIBILITY", "OTHER"],
    "time_perspective": ["PROSPECTIVE", "RETROSPECTIVE", "CROSS_SECTIONAL",
                         "OTHER"],
    "masking": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
    "gender": ["ALL", "FEMALE", "MALE"],
    "lead_sponsor": ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "NETWORK",
                     "INDIV", "OTHER", "UNKNOWN"],
}
DESIGN_FIELDS = ["allocation", "intervention_model", "observational_model",
                 "primary_purpose", "time_perspective", "masking"]
MASKED_ROLES = ["subject_masked", "caregiver_masked", "investigator_masked",
                "outcomes_assessor_masked"]
INTERVENTION_TYPES = ["DRUG", "BIOLOGICAL", "DEVICE", "PROCEDURE",
                      "RADIATION", "BEHAVIORAL", "GENETIC",
                      "DIETARY_SUPPLEMENT", "DIAGNOSTIC_TEST",
                      "COMBINATION_PRODUCT", "OTHER"]
# MeSH top-level disease categories, as they appear in
# browse_conditions.downcase_mesh_term.
THERAPEUTIC_AREAS = [
    "infections", "neoplasms", "musculoskeletal diseases",
    "digestive system diseases", "stomatognathic diseases",
    "respiratory tract diseases", "otorhinolaryngologic diseases",
    "nervous system diseases", "eye diseases", "urogenital diseases",
    "cardiovascular diseases", "hemic and lymphatic diseases",
    "congenital, hereditary, and neonatal diseases and abnormalities",
    "skin and connective tissue diseases",
    "nutritional and metabolic diseases", "endocrine system diseases",
    "immune system diseases", "disorders of environmental origin",
    "occupational diseases", "chemically-induced disorders",
    "wounds and injuries", "pathological conditions, signs and symptoms",
    "mental disorders",
]
# Numeric fields that can be missing; each gets a <name>_missing flag
# before filling with 0.
NULLABLE = ["n_arms", "has_dmc", "is_fda_regulated_drug",
            "is_fda_regulated_device", "start_year", "registration_year",
            "n_facilities", "has_us_facility", "has_single_facility",
            "min_age_years", "max_age_years", "healthy_volunteers", "adult",
            "child", "older_adult"]

STUDY_COLUMNS = ["nct_id", "overall_status", "study_type", "phase",
                 "number_of_arms", "number_of_groups", "has_dmc",
                 "is_fda_regulated_drug", "is_fda_regulated_device",
                 "start_date", "study_first_submitted_date"]
CALCULATED_COLUMNS = ["nct_id", "number_of_facilities", "has_us_facility",
                      "has_single_facility", "minimum_age_num",
                      "minimum_age_unit", "maximum_age_num",
                      "maximum_age_unit",
                      "number_of_primary_outcomes_to_measure",
                      "number_of_secondary_outcomes_to_measure",
                      "number_of_other_outcomes_to_measure"]
ELIGIBILITY_COLUMNS = ["nct_id", "gender", "healthy_volunteers", "adult",
                       "child", "older_adult"]

_AGE_UNIT_YEARS = {"year": 1.0, "month": 1 / 12, "week": 7 / 365.25,
                   "day": 1 / 365.25, "hour": 1 / 8766,
                   "minute": 1 / 525960}


def column_name(text):
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def one_hot(values, prefix, categories):
    """One column per category plus <prefix>_other and <prefix>_missing."""
    v = values.fillna("").astype(str).str.strip()
    cols = {f"{prefix}_{column_name(c)}": (v == c) for c in categories}
    cols[f"{prefix}_other"] = (v != "") & ~v.isin(categories)
    cols[f"{prefix}_missing"] = v == ""
    return pd.DataFrame(cols, index=values.index).astype(np.int8)


def to_flag(values):
    """AACT booleans: 't' -> 1, 'f' -> 0, anything else -> NaN."""
    return values.map({"t": 1.0, "f": 0.0})


def to_number(values):
    return pd.to_numeric(values, errors="coerce")


def age_in_years(num, unit):
    unit = unit.fillna("").astype(str).str.lower().str.strip()
    return to_number(num) * unit.str.rstrip("s").map(_AGE_UNIT_YEARS)


def label_from_status(status):
    """1 = stopped early, 0 = completed, NaN = not finished (censored)."""
    def label(s):
        if s in STOPPED_STATUSES:
            return 1.0
        return 0.0 if s == "COMPLETED" else np.nan
    return status.map(label)


def _first_per_trial(df, index):
    return df.drop_duplicates("nct_id").set_index("nct_id").reindex(index)


def _per_trial(series, index, fill=0):
    return series.reindex(index).fillna(fill)


def _flags_per_trial(present, values, index, prefix, suffix=""):
    """has_<value> int8 columns from a boolean crosstab of nct_id x value."""
    cols = {}
    for value in values:
        col = present[value] if value in present else pd.Series(
            False, index=present.index)
        name = f"{prefix}{column_name(value)}{suffix}"
        cols[name] = _per_trial(col.astype(np.int8), index).astype(np.int8)
    return pd.DataFrame(cols, index=index)


def study_features(studies, index):
    s = _first_per_trial(studies, index)
    f = pd.concat([
        one_hot(s.study_type, "study_type", CATEGORIES["study_type"]),
        one_hot(s.phase, "phase", CATEGORIES["phase"]),
    ], axis=1)
    arms = to_number(s.number_of_arms)
    f["n_arms"] = arms.fillna(to_number(s.number_of_groups))
    for col in ["has_dmc", "is_fda_regulated_drug",
                "is_fda_regulated_device"]:
        f[col] = to_flag(s[col])
    f["start_year"] = to_number(s.start_date.str[:4])
    f["registration_year"] = to_number(s.study_first_submitted_date.str[:4])
    return f


def design_features(designs, index):
    d = _first_per_trial(designs, index)
    f = pd.concat([one_hot(d[c], c, CATEGORIES[c]) for c in DESIGN_FIELDS],
                  axis=1)
    f["n_masked_roles"] = sum((d[r] == "t").astype(np.int8)
                              for r in MASKED_ROLES)
    return f


def calculated_features(calculated, index):
    c = _first_per_trial(calculated, index)
    f = pd.DataFrame(index=index)
    f["n_facilities"] = to_number(c.number_of_facilities)
    f["has_us_facility"] = to_flag(c.has_us_facility)
    f["has_single_facility"] = to_flag(c.has_single_facility)
    f["min_age_years"] = age_in_years(c.minimum_age_num, c.minimum_age_unit)
    f["max_age_years"] = age_in_years(c.maximum_age_num, c.maximum_age_unit)
    for kind in ["primary", "secondary", "other"]:
        planned = c[f"number_of_{kind}_outcomes_to_measure"]
        f[f"n_{kind}_outcomes"] = to_number(planned).fillna(0)
    return f


def criteria_counts(eligibilities):
    """Item counts, word count and split flag from eligibilities.criteria."""
    rows = []
    for criteria in eligibilities.criteria:
        e = split_eligibility(criteria)
        words = sum(len(item.split()) for item in e.inclusion + e.exclusion)
        rows.append((len(e.inclusion), len(e.exclusion), words,
                     int(e.split_found)))
    return pd.DataFrame(rows, index=eligibilities.nct_id.values,
                        columns=["n_inclusion", "n_exclusion",
                                 "criteria_words", "eligibility_split_found"])


def eligibility_features(eligibilities, counts, index):
    e = _first_per_trial(eligibilities, index)
    f = one_hot(e.gender, "gender", CATEGORIES["gender"])
    for col in ["healthy_volunteers", "adult", "child", "older_adult"]:
        f[col] = to_flag(e[col])
    counts = counts[~counts.index.duplicated()].reindex(index)
    return pd.concat([f, counts.fillna(0).astype(np.int64)], axis=1)


def sponsor_features(sponsors, index):
    lead = sponsors[sponsors.lead_or_collaborator == "lead"]
    lead_class = _first_per_trial(lead, index).agency_class
    f = one_hot(lead_class, "lead_sponsor", CATEGORIES["lead_sponsor"])
    collab = sponsors[sponsors.lead_or_collaborator == "collaborator"]
    f["n_collaborators"] = _per_trial(collab.groupby("nct_id").size(), index)
    present = pd.crosstab(collab.nct_id, collab.agency_class) > 0
    flags = _flags_per_trial(present, ["INDUSTRY", "NIH"], index, "has_",
                             "_collaborator")
    return pd.concat([f, flags], axis=1)


def country_features(countries, index):
    # Counts countries ever listed, including ones later removed, to stay
    # close to the original design.
    n = countries.groupby("nct_id").name.nunique()
    return pd.DataFrame({"n_countries": _per_trial(n, index)})


def intervention_features(interventions, index):
    n = interventions.groupby("nct_id").size()
    f = pd.DataFrame({"n_interventions": _per_trial(n, index)})
    present = pd.crosstab(interventions.nct_id,
                          interventions.intervention_type) > 0
    flags = _flags_per_trial(present, INTERVENTION_TYPES, index, "has_",
                             "_intervention")
    return pd.concat([f, flags], axis=1)


def condition_features(conditions, area_rows, index):
    n = conditions.groupby("nct_id").size()
    f = pd.DataFrame({"n_conditions": _per_trial(n, index)})
    present = pd.crosstab(area_rows.nct_id, area_rows.downcase_mesh_term) > 0
    areas = _flags_per_trial(present, THERAPEUTIC_AREAS, index, "area_")
    f = pd.concat([f, areas], axis=1)
    f["n_areas"] = areas.sum(axis=1)
    return f


def fill_missing(features):
    """Add a _missing flag per NULLABLE column, then fill NaN with 0."""
    features = features.copy()
    for col in NULLABLE:
        features[f"{col}_missing"] = features[col].isna().astype(np.int8)
    return features.fillna(0)


def build_features(export_dir):
    def table(name, cols):
        df = pd.concat(read_chunks(export_dir, name, cols),
                       ignore_index=True)
        logger.info("read %s: %d rows", name, len(df))
        return df

    studies = table("studies", STUDY_COLUMNS).drop_duplicates("nct_id")
    index = pd.Index(studies.nct_id, name="nct_id")
    counts = pd.concat(
        criteria_counts(c) for c in read_chunks(
            export_dir, "eligibilities", ["nct_id", "criteria"]))
    logger.info("counted eligibility criteria for %d trials", len(counts))
    area_rows = pd.concat(
        c[c.downcase_mesh_term.isin(THERAPEUTIC_AREAS)]
        for c in read_chunks(export_dir, "browse_conditions",
                             ["nct_id", "downcase_mesh_term"]))

    designs = table("designs", ["nct_id"] + DESIGN_FIELDS + MASKED_ROLES)
    sponsors = table("sponsors",
                     ["nct_id", "agency_class", "lead_or_collaborator"])
    interventions = table("interventions", ["nct_id", "intervention_type"])
    features = pd.concat([
        study_features(studies, index),
        design_features(designs, index),
        calculated_features(table("calculated_values", CALCULATED_COLUMNS),
                            index),
        eligibility_features(table("eligibilities", ELIGIBILITY_COLUMNS),
                             counts, index),
        sponsor_features(sponsors, index),
        country_features(table("countries", ["nct_id", "name"]), index),
        intervention_features(interventions, index),
        condition_features(table("conditions", ["nct_id"]), area_rows,
                           index),
    ], axis=1)
    features = fill_missing(features)

    s = studies.set_index("nct_id")
    ids = pd.DataFrame({
        "overall_status": s.overall_status,
        "label": label_from_status(s.overall_status),
        "start_date": s.start_date,
        "study_first_submitted_date": s.study_first_submitted_date,
    })
    out = pd.concat([ids, features], axis=1).reset_index()

    area_cols = [c for c in out.columns if c.startswith("area_")]
    area_rates = {c: round(float(out[c].mean()), 4) for c in area_cols}
    empty_areas = [c for c in area_cols if out[c].sum() == 0]
    if empty_areas:
        logger.warning("no trials matched therapeutic areas: %s",
                       empty_areas)
    meta = {
        "export_dir": str(export_dir),
        "n_rows": int(len(out)),
        "n_features": int(features.shape[1]),
        "label_counts": {
            "stopped": int((out.label == 1).sum()),
            "completed": int((out.label == 0).sum()),
            "unfinished": int(out.label.isna().sum()),
        },
        "leakage_excluded": LEAKAGE_EXCLUDED,
        "leakage_risk": LEAKAGE_RISK,
        "area_rates": area_rates,
        "feature_columns": list(features.columns),
    }
    return out, meta


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True,
                        help="AACT flat-file export directory")
    parser.add_argument("--out",
                        default="data/processed/structured_features.csv")
    args = parser.parse_args()

    out, meta = build_features(args.export_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.6g")
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %d trials x %d features to %s", meta["n_rows"],
                meta["n_features"], out_path)
    logger.info("labels: %s", meta["label_counts"])


if __name__ == "__main__":
    run_main(main, "structured_features")
