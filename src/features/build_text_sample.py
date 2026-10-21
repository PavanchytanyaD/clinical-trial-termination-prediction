"""Build a cleaned per-trial text table from a local AACT flat-file export.

By default samples trials that have finished (completed or stopped early), for the
BioBERT test run. With --all, cleans every trial in the export.

    python -m src.features.build_text_sample --export-dir <AACT export dir> --n 10000
    python -m src.features.build_text_sample --export-dir <AACT export dir> --all --out data/processed/text_all.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.features.text_cleaning import clean_text, split_eligibility

FINISHED_STATUSES = ["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"]
TEXT_FIELDS = ["title", "brief_summary", "detailed_description", "inclusion", "exclusion", "outcomes"]
_OUTCOME_ORDER = {"primary": 0, "secondary": 1}


def read_chunks(export_dir, table, usecols, nct_ids=None, chunksize=100_000):
    """Yield chunks of selected columns of an AACT pipe-delimited table, optionally filtered to nct_ids."""
    reader = pd.read_csv(Path(export_dir) / f"{table}.txt", sep="|", usecols=usecols,
                         dtype=str, keep_default_na=False, chunksize=chunksize)
    for chunk in reader:
        yield chunk[chunk.nct_id.isin(nct_ids)] if nct_ids is not None else chunk


def clean_column(export_dir, table, column, nct_ids):
    """Cleaned text of one column, indexed by nct_id. Cleans chunk by chunk to bound memory."""
    parts = [pd.Series(c[column].map(clean_text).values, index=c.nct_id)
             for c in read_chunks(export_dir, table, ["nct_id", column], nct_ids)]
    s = pd.concat(parts)
    return s[~s.index.duplicated()]


def split_eligibilities(export_dir, nct_ids):
    """Inclusion/exclusion text, item counts and split flag per trial, indexed by nct_id."""
    parts = []
    for c in read_chunks(export_dir, "eligibilities", ["nct_id", "criteria"], nct_ids):
        rows = []
        for criteria in c.criteria:
            e = split_eligibility(criteria)
            rows.append((e.inclusion_text, e.exclusion_text, len(e.inclusion), len(e.exclusion), e.split_found))
        parts.append(pd.DataFrame(rows, index=c.nct_id.values,
                                  columns=["inclusion", "exclusion", "n_inclusion", "n_exclusion",
                                           "eligibility_split_found"]))
    df = pd.concat(parts)
    return df[~df.index.duplicated()]


def outcomes_texts(export_dir, nct_ids):
    """Planned outcomes (design_outcomes, known at registration), primary first, per trial."""
    by_trial = defaultdict(list)
    cols = ["nct_id", "outcome_type", "measure", "description"]
    for c in read_chunks(export_dir, "design_outcomes", cols, nct_ids):
        for nct_id, kind, measure, description in zip(c.nct_id, c.outcome_type, c.measure, c.description):
            text = clean_text(measure)
            if description:
                text = f"{text}: {clean_text(description)}" if text else clean_text(description)
            if text:
                by_trial[nct_id].append((_OUTCOME_ORDER.get(kind, 2), text))
    return {k: "\n".join(t for _, t in sorted(v, key=lambda p: p[0])) for k, v in by_trial.items()}


def build_sample(export_dir, n, seed, all_trials=False):
    cols = ["nct_id", "overall_status", "brief_title", "official_title"]
    studies = pd.concat(read_chunks(export_dir, "studies", cols), ignore_index=True)
    finished = studies[studies.overall_status.isin(FINISHED_STATUSES)]
    if all_trials:
        sample, ids = studies.reset_index(drop=True), None
    else:
        sample = finished.sample(n=min(n, len(finished)), random_state=seed).reset_index(drop=True)
        ids = set(sample.nct_id)

    out = sample[["nct_id", "overall_status"]].copy()
    titles = sample.official_title.where(sample.official_title != "", sample.brief_title)
    out["title"] = titles.map(clean_text).values
    del studies, finished, sample, titles

    out["brief_summary"] = out.nct_id.map(clean_column(export_dir, "brief_summaries", "description", ids))
    out["detailed_description"] = out.nct_id.map(clean_column(export_dir, "detailed_descriptions", "description", ids))
    elig = split_eligibilities(export_dir, ids)
    for col in elig.columns:
        out[col] = out.nct_id.map(elig[col])
    del elig
    out["outcomes"] = out.nct_id.map(outcomes_texts(export_dir, ids))

    out[TEXT_FIELDS] = out[TEXT_FIELDS].fillna("")
    out[["n_inclusion", "n_exclusion"]] = out[["n_inclusion", "n_exclusion"]].fillna(0).astype(int)
    out["eligibility_split_found"] = out.eligibility_split_found.fillna(False).astype(bool)

    meta = {
        "export_dir": str(export_dir),
        "all_trials": all_trials,
        "n_rows": int(len(out)),
        "seed": None if all_trials else seed,
        "status_counts": out.overall_status.value_counts().to_dict(),
        "eligibility_split_found_rate": float(out.eligibility_split_found.mean()),
        "empty_rate": {f: float((out[f] == "").mean()) for f in TEXT_FIELDS},
    }
    return out, meta


def write_jsonl(df, path, rows_per_write=50_000):
    """Write in slices so large tables do not need one giant JSON string in memory."""
    with open(path, "w", encoding="utf-8") as f:
        for start in range(0, len(df), rows_per_write):
            text = df.iloc[start:start + rows_per_write].to_json(orient="records", lines=True, force_ascii=False)
            f.write(text if text.endswith("\n") else text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, help="AACT flat-file export directory")
    parser.add_argument("--n", type=int, default=10_000, help="sample size (ignored with --all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="clean every trial instead of a sample")
    parser.add_argument("--out", default="data/processed/text_sample_10k.jsonl")
    args = parser.parse_args()

    out, meta = build_sample(args.export_dir, args.n, args.seed, args.all)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, out_path)
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
