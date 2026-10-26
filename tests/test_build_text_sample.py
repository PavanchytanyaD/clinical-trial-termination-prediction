import pandas as pd
import pytest

from src.features.build_text_sample import (
    TEXT_FIELDS, AACTTableError, build_sample, read_chunks, write_jsonl,
)

TABLES = {
    "studies": [
        "nct_id|overall_status|brief_title|official_title",
        "NCT1|COMPLETED|Short one|Official \\>= title",
        "NCT2|TERMINATED|Brief only|",
        "NCT3|RECRUITING|Still running|Running study",
        "NCT4|WITHDRAWN|Never started|Withdrawn study",
    ],
    "brief_summaries": [
        "nct_id|description",
        "NCT1|Line one~Line two", "NCT2|Summary 2", "NCT3|Summary 3",
    ],
    "detailed_descriptions": ["nct_id|description", "NCT1|Details"],
    "eligibilities": [
        "nct_id|criteria",
        '"NCT1"|"Inclusion Criteria:~* Adults~'
        'Exclusion Criteria:~* Pregnancy"',
        "NCT2|Healthy adults",
        "NCT3|Inclusion Criteria:~* Children",
    ],
    "design_outcomes": [
        "nct_id|outcome_type|measure|description",
        "NCT1|secondary|Weight|kg at week 12",
        "NCT1|primary|HbA1c|",
        "NCT1|other||Free text only",
    ],
}


@pytest.fixture
def export_dir(tmp_path):
    for name, lines in TABLES.items():
        path = tmp_path / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_sample_uses_only_finished_trials(export_dir):
    out, meta = build_sample(export_dir, n=10, seed=0)
    assert set(out.nct_id) == {"NCT1", "NCT2", "NCT4"}
    assert meta["n_rows"] == 3 and meta["all_trials"] is False


def test_sample_is_reproducible(export_dir):
    a, _ = build_sample(export_dir, n=2, seed=7)
    b, _ = build_sample(export_dir, n=2, seed=7)
    assert len(a) == 2 and a.nct_id.tolist() == b.nct_id.tolist()


def test_all_trials_and_fields(export_dir):
    out, meta = build_sample(export_dir, n=0, seed=0, all_trials=True)
    out = out.set_index("nct_id")
    assert len(out) == 4 and meta["seed"] is None
    assert out.loc["NCT1", "title"] == "Official >= title"
    assert out.loc["NCT2", "title"] == "Brief only"
    assert out.loc["NCT1", "brief_summary"] == "Line one\nLine two"
    assert out.loc["NCT1", "inclusion"] == "Adults"
    assert out.loc["NCT1", "exclusion"] == "Pregnancy"
    assert not out.loc["NCT2", "eligibility_split_found"]
    assert out.loc["NCT4", "n_inclusion"] == 0


def test_missing_text_becomes_empty(export_dir):
    out, _ = build_sample(export_dir, n=0, seed=0, all_trials=True)
    out = out.set_index("nct_id")
    assert out.loc["NCT4", "brief_summary"] == ""
    assert out.loc["NCT2", "detailed_description"] == ""
    assert not out[TEXT_FIELDS].isna().any().any()


def test_outcomes_primary_first(export_dir):
    out, _ = build_sample(export_dir, n=0, seed=0, all_trials=True)
    outcomes = out.set_index("nct_id").loc["NCT1", "outcomes"].split("\n")
    assert outcomes == ["HbA1c", "Weight: kg at week 12", "Free text only"]


def test_write_jsonl_round_trip(export_dir, tmp_path):
    out, _ = build_sample(export_dir, n=0, seed=0, all_trials=True)
    path = tmp_path / "out.jsonl"
    write_jsonl(out, path, rows_per_write=3)  # forces two slices
    back = pd.read_json(path, lines=True, dtype={"nct_id": str})
    assert back.nct_id.tolist() == out.nct_id.tolist()
    assert back.inclusion.tolist() == out.inclusion.tolist()


def test_read_chunks_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="studies"):
        list(read_chunks(tmp_path, "studies", ["nct_id"]))


def test_read_chunks_missing_column(export_dir):
    with pytest.raises(AACTTableError, match="studies"):
        list(read_chunks(export_dir, "studies", ["nct_id", "no_such_col"]))


def test_read_chunks_corrupt_file(tmp_path):
    (tmp_path / "studies.txt").write_bytes(b"nct_id|x\n\xff\xfe\x00bad|1\n")
    with pytest.raises(AACTTableError):
        list(read_chunks(tmp_path, "studies", ["nct_id", "x"]))
