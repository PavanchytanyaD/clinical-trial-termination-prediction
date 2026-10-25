import numpy as np
import pandas as pd
import pytest

from src.features.structured_features import (
    LEAKAGE_RISK, age_in_years, build_features, label_from_status, one_hot,
    to_flag,
)

TABLES = {
    "studies": [
        "nct_id|overall_status|study_type|phase|number_of_arms|"
        "number_of_groups|has_dmc|is_fda_regulated_drug|"
        "is_fda_regulated_device|start_date|study_first_submitted_date|"
        "enrollment|why_stopped",
        "NCT1|COMPLETED|INTERVENTIONAL|PHASE2|2||t|f||2015-03-01|"
        "2014-12-01|120|",
        "NCT2|WITHDRAWN|INTERVENTIONAL|PHASE1/PHASE2|||||2020-01-01|"
        "2019-11-02|0|No funding",
        "NCT3|RECRUITING|OBSERVATIONAL|NA||3|f|||2024-06-01|2024-05-01|50|",
    ],
    "designs": [
        "nct_id|allocation|intervention_model|observational_model|"
        "primary_purpose|time_perspective|masking|subject_masked|"
        "caregiver_masked|investigator_masked|outcomes_assessor_masked",
        "NCT1|RANDOMIZED|PARALLEL||TREATMENT||DOUBLE|t|f|t|f",
        "NCT3|||COHORT||PROSPECTIVE||||||",
    ],
    "calculated_values": [
        "nct_id|number_of_facilities|has_us_facility|has_single_facility|"
        "minimum_age_num|minimum_age_unit|maximum_age_num|maximum_age_unit|"
        "number_of_primary_outcomes_to_measure|"
        "number_of_secondary_outcomes_to_measure|"
        "number_of_other_outcomes_to_measure",
        "NCT1|12|t|f|18|year|||1|3|",
        "NCT2|0|||6|month|17|years|1||",
    ],
    "eligibilities": [
        "nct_id|gender|healthy_volunteers|adult|child|older_adult|criteria",
        'NCT1|ALL|f|t|f|t|"Inclusion Criteria:~* Adults~* Type 2 diabetes~'
        'Exclusion Criteria:~* Pregnancy"',
        "NCT2|FEMALE|t|f|t|f|Healthy girls",
    ],
    "sponsors": [
        "nct_id|agency_class|lead_or_collaborator",
        "NCT1|INDUSTRY|lead", "NCT1|NIH|collaborator",
        "NCT1|OTHER|collaborator", "NCT2|OTHER|lead",
    ],
    "countries": ["nct_id|name", "NCT1|United States", "NCT1|Canada",
                  "NCT2|France"],
    "interventions": ["nct_id|intervention_type", "NCT1|DRUG", "NCT1|DRUG",
                      "NCT2|BEHAVIORAL"],
    "conditions": ["nct_id", "NCT1", "NCT2", "NCT2"],
    "browse_conditions": [
        "nct_id|downcase_mesh_term",
        "NCT1|nutritional and metabolic diseases",
        "NCT1|endocrine system diseases", "NCT1|diabetes mellitus",
    ],
}


@pytest.fixture
def export_dir(tmp_path):
    for name, lines in TABLES.items():
        path = tmp_path / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def built(export_dir):
    out, meta = build_features(export_dir)
    return out.set_index("nct_id"), meta


def test_label_from_status():
    statuses = ["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED",
                "RECRUITING"]
    labels = label_from_status(pd.Series(statuses))
    assert labels[:4].tolist() == [0, 1, 1, 1]
    assert np.isnan(labels[4])


def test_one_hot_fixed_columns():
    f = one_hot(pd.Series(["A", "", "Z", None]), "x", ["A", "B"])
    assert list(f.columns) == ["x_a", "x_b", "x_unlisted", "x_missing"]
    assert f.values.tolist() == [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                                 [0, 0, 0, 1]]


def test_to_flag_and_age():
    assert to_flag(pd.Series(["t", "f", ""])).tolist()[:2] == [1.0, 0.0]
    ages = age_in_years(pd.Series(["18", "6", "", "2"]),
                        pd.Series(["year", "Months", "year", "weeks"]))
    assert ages[0] == 18
    assert ages[1] == pytest.approx(0.5)
    assert np.isnan(ages[2])
    assert ages[3] == pytest.approx(14 / 365.25)


def test_one_row_per_trial_with_label(built):
    out, meta = built
    assert list(out.index) == ["NCT1", "NCT2", "NCT3"]
    assert out.label.tolist()[:2] == [0, 1]
    assert np.isnan(out.label["NCT3"])
    assert meta["label_counts"] == {"stopped": 1, "completed": 1,
                                    "unfinished": 1}


def test_no_leakage_columns(built):
    out, _ = built
    for col in ["enrollment", "why_stopped", "enrollment_type",
                "completion_date"]:
        assert col not in out.columns


def test_no_missing_values_in_features(built):
    out, meta = built
    assert not out[meta["feature_columns"]].isna().any().any()
    assert all(c in meta["feature_columns"] for c in LEAKAGE_RISK)


def test_design_and_study_features(built):
    out, _ = built
    assert out.loc["NCT1", "phase_phase2"] == 1
    assert out.loc["NCT2", "phase_phase1_phase2"] == 1
    assert out.loc["NCT3", "phase_na"] == 1
    assert out.loc["NCT1", "n_arms"] == 2
    assert out.loc["NCT3", "n_arms"] == 3
    assert out.loc["NCT2", "n_arms_missing"] == 1
    assert out.loc["NCT1", "masking_double"] == 1
    assert out.loc["NCT1", "n_masked_roles"] == 2
    assert out.loc["NCT2", "masking_missing"] == 1  # no designs row
    assert out.loc["NCT1", "start_year"] == 2015


def test_age_eligibility_and_counts(built):
    out, _ = built
    assert out.loc["NCT2", "min_age_years"] == pytest.approx(0.5)
    assert out.loc["NCT2", "max_age_years"] == 17
    assert out.loc["NCT1", "max_age_years_missing"] == 1
    assert out.loc["NCT1", "n_inclusion"] == 2
    assert out.loc["NCT1", "n_exclusion"] == 1
    assert out.loc["NCT2", "eligibility_split_found"] == 0
    assert out.loc["NCT2", "gender_female"] == 1
    assert out.loc["NCT3", "gender_missing"] == 1


def test_sponsor_country_intervention_condition(built):
    out, _ = built
    assert out.loc["NCT1", "lead_sponsor_industry"] == 1
    assert out.loc["NCT1", "n_collaborators"] == 2
    assert out.loc["NCT1", "has_nih_collaborator"] == 1
    assert out.loc["NCT2", "has_nih_collaborator"] == 0
    assert out.loc["NCT1", "n_countries"] == 2
    assert out.loc["NCT3", "n_countries"] == 0
    assert out.loc["NCT1", "has_drug_intervention"] == 1
    assert out.loc["NCT1", "n_interventions"] == 2
    assert out.loc["NCT2", "has_behavioral_intervention"] == 1
    assert out.loc["NCT1", "area_endocrine_system_diseases"] == 1
    assert out.loc["NCT1", "n_areas"] == 2
    assert out.loc["NCT2", "n_conditions"] == 2


def test_real_other_category_is_kept(built):
    out, _ = built
    # NCT2's lead sponsor class is OTHER; it must not be overwritten by the
    # catch-all column for values outside the list
    assert out.loc["NCT2", "lead_sponsor_other"] == 1
    assert out.loc["NCT2", "lead_sponsor_unlisted"] == 0


def test_facility_missing_flags_are_leakage_risk(built):
    _, meta = built
    for col in ["n_facilities_missing", "has_us_facility_missing",
                "has_single_facility_missing"]:
        assert col in LEAKAGE_RISK and col in meta["feature_columns"]


def test_empty_snapshot_is_an_error(export_dir):
    (export_dir / "studies.txt").write_text(TABLES["studies"][0] + "\n")
    with pytest.raises(ValueError, match="no trials"):
        build_features(export_dir)


def test_unknown_status_format_is_an_error(export_dir):
    lines = [TABLES["studies"][0]] + [
        line.replace("COMPLETED", "Completed").replace("WITHDRAWN",
                                                       "Withdrawn")
        for line in TABLES["studies"][1:]]
    (export_dir / "studies.txt").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="overall_status format"):
        build_features(export_dir)
