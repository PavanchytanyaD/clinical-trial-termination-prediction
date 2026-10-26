"""Build a cleaned per-trial text table from an AACT flat-file export.

The pipeline's text stage calls build_sample(..., all_trials=True), which
cleans every trial. By default the command line samples trials that have
finished (completed or stopped early); --all cleans every trial.

    python -m src.features.build_text_sample --export-dir <dir> --n 10000
    python -m src.features.build_text_sample --export-dir <dir> --all \\
        --out data/processed/text_all.jsonl
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.error_handling import DataUnavailableError, DataValidationError
from src.features.text_cleaning import clean_text, split_eligibility
from src.logging_utils import run_main

logger = logging.getLogger(__name__)

FINISHED_STATUSES = ["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"]
TEXT_FIELDS = ["title", "brief_summary", "detailed_description",
               "inclusion", "exclusion", "outcomes"]
ELIGIBILITY_COLUMNS = ["inclusion", "exclusion", "n_inclusion",
                       "n_exclusion", "eligibility_split_found"]
_OUTCOME_ORDER = {"primary": 0, "secondary": 1}


class AACTTableError(DataValidationError, RuntimeError):
    """An AACT table file exists but cannot be read."""


def read_chunks(export_dir, table, usecols, nct_ids=None, chunksize=100_000):
    """Yield chunks of selected columns of an AACT pipe-delimited table.

    Rows are filtered to nct_ids when given. Raises FileNotFoundError for a
    missing table and AACTTableError for one that cannot be parsed (corrupt
    file or missing columns).
    """
    path = Path(export_dir) / f"{table}.txt"
    if not path.exists():
        raise DataUnavailableError(f"AACT table not found: {path}")
    try:
        reader = pd.read_csv(path, sep="|", usecols=usecols, dtype=str,
                             keep_default_na=False, chunksize=chunksize)
        for chunk in reader:
            if nct_ids is not None:
                chunk = chunk[chunk.nct_id.isin(nct_ids)]
            yield chunk
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        raise AACTTableError(f"could not read {path}: {e}") from e


def clean_column(export_dir, table, column, nct_ids):
    """Cleaned text of one column, indexed by nct_id (cleaned per chunk)."""
    parts = [pd.Series(c[column].map(clean_text).values, index=c.nct_id)
             for c in read_chunks(export_dir, table, ["nct_id", column],
                                  nct_ids)]
    s = pd.concat(parts)
    logger.info("cleaned %s.%s for %d trials", table, column, len(s))
    return s[~s.index.duplicated()]


def split_eligibilities(export_dir, nct_ids):
    """Inclusion/exclusion text, item counts and split flag per trial."""
    parts = []
    cols = ["nct_id", "criteria"]
    for c in read_chunks(export_dir, "eligibilities", cols, nct_ids):
        rows = []
        for criteria in c.criteria:
            e = split_eligibility(criteria)
            rows.append((e.inclusion_text, e.exclusion_text,
                         len(e.inclusion), len(e.exclusion), e.split_found))
        parts.append(pd.DataFrame(rows, index=c.nct_id.values,
                                  columns=ELIGIBILITY_COLUMNS))
    df = pd.concat(parts)
    logger.info("split eligibility criteria for %d trials", len(df))
    return df[~df.index.duplicated()]


def outcomes_texts(export_dir, nct_ids):
    """Planned outcomes per trial (design_outcomes), primary first.

    These are the outcome measures written at registration, not results.
    """
    by_trial = defaultdict(list)
    cols = ["nct_id", "outcome_type", "measure", "description"]
    for c in read_chunks(export_dir, "design_outcomes", cols, nct_ids):
        rows = zip(c.nct_id, c.outcome_type, c.measure, c.description)
        for nct_id, kind, measure, description in rows:
            text = clean_text(measure)
            if description:
                desc = clean_text(description)
                text = f"{text}: {desc}" if text else desc
            if text:
                order = _OUTCOME_ORDER.get(str(kind).lower(), 2)
                by_trial[nct_id].append((order, text))
    logger.info("built planned outcomes for %d trials", len(by_trial))
    return {k: "\n".join(t for _, t in sorted(v, key=lambda p: p[0]))
            for k, v in by_trial.items()}


def build_sample(export_dir, n, seed, all_trials=False):
    cols = ["nct_id", "overall_status", "brief_title", "official_title"]
    studies = pd.concat(read_chunks(export_dir, "studies", cols),
                        ignore_index=True)
    finished = studies[studies.overall_status.isin(FINISHED_STATUSES)]
    logger.info("%d studies, %d finished", len(studies), len(finished))
    if all_trials:
        sample, ids = studies.reset_index(drop=True), None
    else:
        if n > len(finished):
            logger.warning("asked for %d trials but only %d are finished",
                           n, len(finished))
        sample = finished.sample(n=min(n, len(finished)), random_state=seed)
        sample = sample.reset_index(drop=True)
        ids = set(sample.nct_id)

    out = sample[["nct_id", "overall_status"]].copy()
    titles = sample.official_title.where(sample.official_title != "",
                                         sample.brief_title)
    out["title"] = titles.map(clean_text).values
    del studies, finished, sample, titles

    for field_name, table in [("brief_summary", "brief_summaries"),
                              ("detailed_description",
                               "detailed_descriptions")]:
        text = clean_column(export_dir, table, "description", ids)
        out[field_name] = out.nct_id.map(text)
    elig = split_eligibilities(export_dir, ids)
    for col in elig.columns:
        out[col] = out.nct_id.map(elig[col])
    del elig
    out["outcomes"] = out.nct_id.map(outcomes_texts(export_dir, ids))

    out[TEXT_FIELDS] = out[TEXT_FIELDS].fillna("")
    counts = ["n_inclusion", "n_exclusion"]
    out[counts] = out[counts].fillna(0).astype(int)
    out["eligibility_split_found"] = (
        out.eligibility_split_found.fillna(False).astype(bool))

    meta = {
        "export_dir": str(export_dir),
        "all_trials": all_trials,
        "n_rows": int(len(out)),
        "seed": None if all_trials else seed,
        "status_counts": out.overall_status.value_counts().to_dict(),
        "eligibility_split_found_rate":
            float(out.eligibility_split_found.mean()),
        "empty_rate": {f: float((out[f] == "").mean()) for f in TEXT_FIELDS},
    }
    return out, meta


def write_jsonl(df, path, rows_per_write=50_000):
    """Write in slices so large tables never become one huge JSON string."""
    with open(path, "w", encoding="utf-8") as f:
        for start in range(0, len(df), rows_per_write):
            part = df.iloc[start:start + rows_per_write]
            text = part.to_json(orient="records", lines=True,
                                force_ascii=False)
            f.write(text if text.endswith("\n") else text + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True,
                        help="AACT flat-file export directory")
    parser.add_argument("--n", type=int, default=10_000,
                        help="sample size (ignored with --all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true",
                        help="clean every trial instead of a sample")
    parser.add_argument("--out",
                        default="data/processed/text_sample_10k.jsonl")
    args = parser.parse_args()

    out, meta = build_sample(args.export_dir, args.n, args.seed, args.all)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, out_path)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %d trials to %s", len(out), out_path)
    logger.info("summary: %s", json.dumps(meta))


if __name__ == "__main__":
    run_main(main, "build_text_sample")
