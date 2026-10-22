"""BioBERT test run on the text sample.

Reports token lengths and embedding time per field and saves the embeddings.

    python -m src.features.biobert_test_run \\
        --sample data/processed/text_sample_10k.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.biobert_embeddings import (
    MODEL_NAME, embed_texts, load_biobert,
)
from src.features.build_text_sample import TEXT_FIELDS
from src.logging_utils import run_main

logger = logging.getLogger(__name__)


def token_stats(texts, tokenizer, max_length):
    lengths = np.array([len(tokenizer(t)["input_ids"]) if t else 0
                        for t in texts])
    nonempty = lengths[lengths > 0]

    def pct(q):
        return int(np.percentile(nonempty, q)) if len(nonempty) else 0

    return {
        "empty_rate": float((lengths == 0).mean()),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": int(nonempty.max()) if len(nonempty) else 0,
        "over_max_length_rate": float((lengths > max_length).mean()),
    }


def load_sample(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"text sample not found: {path} (run build_text_sample first)")
    df = pd.read_json(path, lines=True, dtype={"nct_id": str})
    missing = [f for f in TEXT_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing text fields: {missing}")
    df[TEXT_FIELDS] = df[TEXT_FIELDS].fillna("")
    return df


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample",
                        default="data/processed/text_sample_10k.jsonl")
    parser.add_argument("--out-dir",
                        default="data/processed/biobert_test_run")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="cap windows per text (default: no cap)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = load_sample(args.sample)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_biobert(args.model)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    report = {"model": args.model, "device": device, "gpu": gpu,
              "n_trials": len(df), "max_length": args.max_length,
              "stride": args.stride, "max_chunks": args.max_chunks,
              "batch_size": args.batch_size, "fields": {}}

    total_windows, total_seconds = 0, 0.0
    for name in TEXT_FIELDS:
        texts = df[name].tolist()
        stats = token_stats(texts, tokenizer, args.max_length)
        _sync(device)
        start = time.perf_counter()
        vecs, n_windows = embed_texts(
            texts, tokenizer, model, device, args.max_length, args.stride,
            args.max_chunks, args.batch_size)
        _sync(device)
        seconds = time.perf_counter() - start
        np.save(out_dir / f"embeddings_{name}.npy", vecs)
        stats.update({
            "windows": int(n_windows.sum()),
            "mean_windows_per_text": float(n_windows.mean()),
            "max_windows_per_text": int(n_windows.max()),
            "seconds": round(seconds, 1),
            "nan_rows": int(np.isnan(vecs).any(axis=1).sum()),
        })
        if stats["nan_rows"]:
            logger.warning("%s: %d embeddings contain NaN", name,
                           stats["nan_rows"])
        report["fields"][name] = stats
        total_windows += stats["windows"]
        total_seconds += seconds
        logger.info("%s: windows=%d over %d tokens=%.1f%% p90=%d %.1fs",
                    name, stats["windows"], args.max_length,
                    100 * stats["over_max_length_rate"], stats["p90"],
                    seconds)

    (out_dir / "nct_ids.txt").write_text("\n".join(df.nct_id))
    report["total_windows"] = total_windows
    report["total_minutes"] = round(total_seconds / 60, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("total windows=%d time=%.1f min", total_windows,
                total_seconds / 60)


if __name__ == "__main__":
    run_main(main, "biobert_test_run")
