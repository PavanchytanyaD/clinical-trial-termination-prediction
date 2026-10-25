"""Placeholder data acquisition: write a small synthetic AACT snapshot.

The snapshot has the same 12 pipe-delimited tables and columns as the real
AACT flat-file export (only the columns the project reads), so every later
step (preprocessing, schema, statistics, validation, versioning) runs
end to end without an AACT account. Values are random but seeded, so the
same parameters always give the same files.

Real data replaces this through acquire.source = zip or db in params.yaml,
which calls src/ingestion/fetch_aact.py.
"""

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

STATUSES = (["COMPLETED"] * 55 + ["TERMINATED"] * 12 + ["WITHDRAWN"] * 5
            + ["SUSPENDED"] * 1 + ["RECRUITING"] * 20
            + ["ACTIVE_NOT_RECRUITING"] * 7)
STOPPED = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
PHASES = ["NA", "EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2", "PHASE2",
          "PHASE2/PHASE3", "PHASE3", "PHASE4", ""]
SPONSOR_CLASSES = ["INDUSTRY", "OTHER", "NIH", "OTHER_GOV", "NETWORK"]
INTERVENTION_TYPES = ["DRUG", "BIOLOGICAL", "DEVICE", "PROCEDURE",
                      "BEHAVIORAL", "DIETARY_SUPPLEMENT", "OTHER"]
AREAS = ["neoplasms", "cardiovascular diseases", "nervous system diseases",
         "infections", "mental disorders",
         "nutritional and metabolic diseases"]
COUNTRIES = ["United States", "Canada", "Germany", "France", "China",
             "India", "Brazil", "United Kingdom"]
WHY_STOPPED = ["Slow accrual", "Sponsor decision", "Lack of funding",
               "Safety concerns", "PI left institution"]

COLUMNS = {
    "studies": ["nct_id", "overall_status", "study_type", "phase",
                "number_of_arms", "number_of_groups", "has_dmc",
                "is_fda_regulated_drug", "is_fda_regulated_device",
                "start_date", "study_first_submitted_date", "brief_title",
                "official_title", "why_stopped"],
    "brief_summaries": ["id", "nct_id", "description"],
    "detailed_descriptions": ["id", "nct_id", "description"],
    "eligibilities": ["id", "nct_id", "gender", "minimum_age", "maximum_age",
                      "healthy_volunteers", "criteria", "adult", "child",
                      "older_adult"],
    "design_outcomes": ["id", "nct_id", "outcome_type", "measure",
                        "time_frame", "description"],
    "designs": ["id", "nct_id", "allocation", "intervention_model",
                "observational_model", "primary_purpose", "time_perspective",
                "masking", "subject_masked", "caregiver_masked",
                "investigator_masked", "outcomes_assessor_masked"],
    "calculated_values": ["id", "nct_id", "number_of_facilities",
                          "has_us_facility", "has_single_facility",
                          "minimum_age_num", "minimum_age_unit",
                          "maximum_age_num", "maximum_age_unit",
                          "number_of_primary_outcomes_to_measure",
                          "number_of_secondary_outcomes_to_measure",
                          "number_of_other_outcomes_to_measure"],
    "sponsors": ["id", "nct_id", "agency_class", "lead_or_collaborator",
                 "name"],
    "countries": ["id", "nct_id", "name", "removed"],
    "interventions": ["id", "nct_id", "intervention_type", "name"],
    "conditions": ["id", "nct_id", "name", "downcase_name"],
    "browse_conditions": ["id", "nct_id", "mesh_term", "downcase_mesh_term",
                          "mesh_type"],
}


def _flag(rng, p_true, p_missing=0.05):
    r = rng.random()
    return "" if r < p_missing else ("t" if r < p_missing + p_true else "f")


def _date(rng, first_year, last_year):
    day = dt.date(first_year, 1, 1) + dt.timedelta(
        days=int(rng.integers(0, (last_year - first_year + 1) * 365)))
    return day.isoformat() if rng.random() > 0.1 else day.isoformat()[:7]


def generate_tables(n_trials=500, seed=42):
    """Return {table: list of row dicts} for n_trials synthetic trials."""
    rng = np.random.default_rng(seed)
    rows = {t: [] for t in COLUMNS}
    ids = {t: 0 for t in COLUMNS}

    def add(table, **values):
        ids[table] += 1
        rows[table].append({"id": ids[table], **values})

    for i in range(n_trials):
        nct = f"NCT{10000000 + i * 7919 % 89999999:08d}"
        status = STATUSES[int(rng.integers(len(STATUSES)))]
        stopped = status in STOPPED
        interventional = rng.random() < 0.78
        n_arms = int(rng.integers(1, 5))
        start = _date(rng, 2000, 2025)
        submitted = _date(rng, 1999, 2025)
        rows["studies"].append({
            "nct_id": nct, "overall_status": status,
            "study_type": "INTERVENTIONAL" if interventional
            else "OBSERVATIONAL",
            "phase": PHASES[int(rng.integers(len(PHASES)))] if interventional
            else "NA",
            "number_of_arms": n_arms if interventional else "",
            "number_of_groups": "" if interventional else n_arms,
            "has_dmc": _flag(rng, 0.4),
            "is_fda_regulated_drug": _flag(rng, 0.3),
            "is_fda_regulated_device": _flag(rng, 0.1),
            "start_date": start, "study_first_submitted_date": submitted,
            "brief_title": f"Study {i} of an investigational treatment",
            "official_title": f"A Phase Study {i} Evaluating Safety and "
                              "Efficacy" if rng.random() > 0.2 else "",
            "why_stopped": WHY_STOPPED[int(rng.integers(len(WHY_STOPPED)))]
            if stopped else "",
        })
        add("brief_summaries", nct_id=nct,
            description=f"This study evaluates treatment {i}.~Participants "
                        "receive the study drug or placebo.")
        if rng.random() < 0.7:
            add("detailed_descriptions", nct_id=nct,
                description="Detailed description of the design.~~"
                            "Visits occur every 4 weeks.")
        n_incl = int(rng.integers(1, 8))
        n_excl = int(rng.integers(1, 12)) + (4 if stopped else 0)
        criteria = ("Inclusion Criteria:~~"
                    + "~".join(f"* inclusion item {k}" for k in range(n_incl))
                    + "~~Exclusion Criteria:~~"
                    + "~".join(f"* exclusion item {k}" for k in range(n_excl)))
        min_age = int(rng.choice([0, 12, 18, 18, 18, 40, 65]))
        max_age = int(rng.choice([a for a in (17, 45, 65, 75, 99)
                                  if a > min_age]))
        add("eligibilities", nct_id=nct,
            gender=rng.choice(["ALL", "ALL", "ALL", "FEMALE", "MALE"]),
            minimum_age=f"{min_age} Years" if min_age else "",
            maximum_age=f"{max_age} Years", healthy_volunteers=_flag(rng, 0.2),
            criteria=criteria, adult="t", child="t" if min_age < 18 else "f",
            older_adult="t" if max_age >= 65 else "f")
        for kind, n in [("PRIMARY", int(rng.integers(1, 3))),
                        ("SECONDARY", int(rng.integers(0, 5)))]:
            for k in range(n):
                add("design_outcomes", nct_id=nct, outcome_type=kind,
                    measure=f"{kind.title()} outcome {k}",
                    time_frame="12 weeks", description="Change from baseline")
        add("designs", nct_id=nct,
            allocation=rng.choice(["RANDOMIZED", "NON_RANDOMIZED", "NA"])
            if interventional else "",
            intervention_model=rng.choice(["PARALLEL", "SINGLE_GROUP",
                                           "CROSSOVER"])
            if interventional else "",
            observational_model="" if interventional
            else rng.choice(["COHORT", "CASE_CONTROL"]),
            primary_purpose=rng.choice(["TREATMENT", "PREVENTION",
                                        "SUPPORTIVE_CARE", "DIAGNOSTIC"])
            if interventional else "",
            time_perspective="" if interventional else "PROSPECTIVE",
            masking=rng.choice(["NONE", "SINGLE", "DOUBLE", "QUADRUPLE"])
            if interventional else "",
            subject_masked=_flag(rng, 0.4), caregiver_masked=_flag(rng, 0.2),
            investigator_masked=_flag(rng, 0.3),
            outcomes_assessor_masked=_flag(rng, 0.3))
        n_fac = int(rng.integers(1, 6 if stopped else 40))
        add("calculated_values", nct_id=nct, number_of_facilities=n_fac,
            has_us_facility=_flag(rng, 0.45), has_single_facility=
            "t" if n_fac == 1 else "f",
            minimum_age_num=min_age or "", minimum_age_unit="Years"
            if min_age else "", maximum_age_num=max_age,
            maximum_age_unit="Years",
            number_of_primary_outcomes_to_measure=int(rng.integers(1, 3)),
            number_of_secondary_outcomes_to_measure=int(rng.integers(0, 6)),
            number_of_other_outcomes_to_measure=int(rng.integers(0, 2)))
        add("sponsors", nct_id=nct, lead_or_collaborator="lead",
            agency_class=rng.choice(SPONSOR_CLASSES), name=f"Sponsor {i % 37}")
        for k in range(int(rng.integers(0, 3))):
            add("sponsors", nct_id=nct, lead_or_collaborator="collaborator",
                agency_class=rng.choice(SPONSOR_CLASSES),
                name=f"Collaborator {k}")
        for c in rng.choice(COUNTRIES, size=int(rng.integers(1, 4)),
                            replace=False):
            add("countries", nct_id=nct, name=c, removed=_flag(rng, 0.05, 0))
        for k in range(int(rng.integers(1, 4))):
            add("interventions", nct_id=nct,
                intervention_type=rng.choice(INTERVENTION_TYPES),
                name=f"Intervention {k}")
        area = AREAS[int(rng.integers(len(AREAS)))]
        add("conditions", nct_id=nct, name=f"Condition in {area}",
            downcase_name=f"condition in {area}")
        add("browse_conditions", nct_id=nct, mesh_term=area.title(),
            downcase_mesh_term=area, mesh_type="mesh-ancestor")
    return rows


def write_snapshot(out_dir, snapshot, n_trials=500, seed=42):
    """Write the tables and a manifest.json; returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for table, table_rows in generate_tables(n_trials, seed).items():
        path = out_dir / f"{table}.txt"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS[table],
                                    delimiter="|", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table_rows)
        sizes[table] = path.stat().st_size
    manifest = {
        "source": "sample",
        "snapshot": str(snapshot),
        "n_trials": n_trials,
        "seed": seed,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "tables": sizes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
