"""BioBERT text embeddings (model used as is, not fine-tuned).

Texts longer than max_length tokens are split into overlapping windows. Each
window is mean-pooled over its tokens, and a text's embedding is the
token-weighted mean of its windows. Empty texts get a zero vector.
"""

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


def load_biobert(model_name=MODEL_NAME, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # The model is cased; set it explicitly so a missing tokenizer
        # config cannot lowercase the input.
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  do_lower_case=False)
        model = AutoModel.from_pretrained(model_name)
    except OSError as e:
        raise RuntimeError(
            f"could not load {model_name}; check the name and that Hugging "
            f"Face is reachable (or the model is cached): {e}") from e
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    logger.info("loaded %s on %s", model_name, device)
    return tokenizer, model, device


def chunk_texts(texts, tokenizer, max_length=512, stride=64,
                max_chunks=None):
    """Tokenize texts into windows. Returns (windows, owner per window)."""
    windows, owners = [], []
    for i, text in enumerate(texts):
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        stride=stride, return_overflowing_tokens=True)
        ids = enc["input_ids"]
        if max_chunks:
            ids = ids[:max_chunks]
        windows.extend(ids)
        owners.extend([i] * len(ids))
    return windows, np.array(owners, dtype=np.int64)


def _pool_batch(windows, idx, counts, model, device, pad_id):
    """Mean-pooled vectors for the windows in idx (one padded batch)."""
    width = int(counts[idx].max())
    ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    for row, j in enumerate(idx):
        ids[row, :len(windows[j])] = torch.tensor(windows[j])
    ids = ids.to(device)
    mask = (ids != pad_id).unsqueeze(-1)
    out = model(input_ids=ids,
                attention_mask=mask.squeeze(-1).long()).last_hidden_state
    pooled = (out * mask).sum(1) / mask.sum(1)
    return pooled.float().cpu().numpy()


def _embed_batch(windows, idx, counts, model, device, pad_id, vecs):
    """Embed one batch; on GPU out-of-memory, split it in half and retry."""
    try:
        vecs[idx] = _pool_batch(windows, idx, counts, model, device, pad_id)
    except torch.cuda.OutOfMemoryError:
        if len(idx) == 1:
            raise
        logger.warning("GPU out of memory on a batch of %d; splitting it",
                       len(idx))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = len(idx) // 2
        for part in (idx[:half], idx[half:]):
            _embed_batch(windows, part, counts, model, device, pad_id, vecs)


@torch.no_grad()
def embed_windows(windows, model, device, pad_id, batch_size=32):
    """Mean-pooled embedding and token count for each window."""
    vecs = np.zeros((len(windows), model.config.hidden_size),
                    dtype=np.float32)
    counts = np.array([len(w) for w in windows], dtype=np.float32)
    # Batch windows of similar length together to minimize padding.
    order = np.argsort(counts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        _embed_batch(windows, idx, counts, model, device, pad_id, vecs)
    return vecs, counts


def embed_texts(texts, tokenizer, model, device, max_length=512, stride=64,
                max_chunks=None, batch_size=32):
    """Return an (n_texts, hidden) float32 array and windows per text."""
    windows, owners = chunk_texts(texts, tokenizer, max_length, stride,
                                  max_chunks)
    hidden = model.config.hidden_size
    out = np.zeros((len(texts), hidden), dtype=np.float32)
    n_windows = np.bincount(owners, minlength=len(texts))
    if not windows:
        return out, n_windows
    vecs, counts = embed_windows(windows, model, device,
                                 tokenizer.pad_token_id, batch_size)
    np.add.at(out, owners, vecs * counts[:, None])
    weights = np.bincount(owners, weights=counts, minlength=len(texts))
    nonzero = weights > 0
    out[nonzero] /= weights[nonzero, None]
    return out, n_windows
