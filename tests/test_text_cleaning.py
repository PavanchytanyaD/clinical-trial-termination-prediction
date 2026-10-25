from src.features.text_cleaning import clean_text, split_eligibility


def test_clean_text_handles_aact_encoding():
    raw = "BMI \\>=25 and age \\<65~~  deﬁned   as  Phase Ⅲ\x07"
    assert clean_text(raw) == "BMI >=25 and age <65\ndefined as Phase III"


def test_clean_text_missing():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_clean_text_drops_boilerplate():
    raw = ("Adults with asthma~"
           "* Other protocol-defined inclusion criteria may apply.")
    assert clean_text(raw) == "Adults with asthma"


def test_split_standard():
    raw = ("Inclusion Criteria:~* Age 18 or older~* Signed consent~"
           "Exclusion Criteria:~* Pregnancy~* eGFR \\< 30")
    e = split_eligibility(raw)
    assert e.split_found
    assert e.inclusion == ["Age 18 or older", "Signed consent"]
    assert e.exclusion == ["Pregnancy", "eGFR < 30"]


def test_split_header_variants():
    raw = ("Key Inclusion Criteria~1. Adults~2. ECOG 0-1~"
           "Non-inclusion criteria：~a) Prior chemotherapy")
    e = split_eligibility(raw)
    assert e.inclusion == ["Adults", "ECOG 0-1"]
    assert e.exclusion == ["Prior chemotherapy"]


def test_split_header_with_inline_tail_and_none():
    e = split_eligibility(
        "Inclusion: healthy volunteers~Exclusion criteria: none")
    assert e.inclusion == ["healthy volunteers"]
    assert e.exclusion == []


def test_split_preamble_goes_to_inclusion():
    e = split_eligibility("Adults with COPD~Exclusion Criteria:~Asthma")
    assert e.inclusion == ["Adults with COPD"]
    assert e.exclusion == ["Asthma"]


def test_split_no_headers():
    e = split_eligibility("Healthy adults aged 18-45")
    assert not e.split_found
    assert e.inclusion == ["Healthy adults aged 18-45"]


def test_criterion_mentioning_exclusion_is_not_a_header():
    raw = ("Inclusion Criteria:~* Patients who met all exclusion criteria "
           "of the parent study and completed follow-up")
    e = split_eligibility(raw)
    assert e.exclusion == []
    assert len(e.inclusion) == 1


def test_bullets_stripped_but_numbers_and_operators_kept():
    e = split_eligibility(
        "Inclusion Criteria:~1. Age 18 or older~2)Signed consent~"
        "* Adults~1.5 x ULN or lower~\\>= 18 years~-20 C storage")
    assert e.inclusion == ["Age 18 or older", "Signed consent", "Adults",
                           "1.5 x ULN or lower", ">= 18 years",
                           "-20 C storage"]
