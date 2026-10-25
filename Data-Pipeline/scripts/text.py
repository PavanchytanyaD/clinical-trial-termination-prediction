"""Pipeline stage: cleaned free text for every trial (input to BioBERT).

Calls src/features/build_text_sample.py (Task 1) on the snapshot and writes
one JSON line per trial to data/processed/text_all.jsonl, with the fields
title, brief_summary, detailed_description, inclusion, exclusion and
outcomes (registration-time text only; why_stopped and results are not
used), plus text_all.meta.json with row counts and empty rates.

    python -m scripts.text
"""

import json
import logging

from scripts.config import PROCESSED_DIR, RAW_DIR
from scripts.tracking import fingerprint, tracked_step

logger = logging.getLogger("text")

TEXT_JSONL = PROCESSED_DIR / "text_all.jsonl"
TEXT_META = PROCESSED_DIR / "text_all.meta.json"


def main():
    from src.features.build_text_sample import build_sample, write_jsonl

    with tracked_step("text") as run:
        out, meta = build_sample(RAW_DIR, n=0, seed=0, all_trials=True)
        TEXT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out, TEXT_JSONL)
        TEXT_META.write_text(json.dumps(meta, indent=2))
        run["metrics"].update({
            "rows": meta["n_rows"],
            **{f"empty_{k}": round(v, 4)
               for k, v in meta["empty_rate"].items()}})
        run["outputs"]["data/processed/text_all.jsonl"] = \
            fingerprint(TEXT_JSONL)
        logger.info("wrote cleaned text for %d trials to %s",
                    meta["n_rows"], TEXT_JSONL)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1)
