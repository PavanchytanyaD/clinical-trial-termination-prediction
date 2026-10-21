"""BioBERT text embeddings (model used as is, not fine-tuned).

Texts longer than max_length tokens are split into overlapping windows. Each window is
mean-pooled over its tokens, and a text's embedding is the token-weighted mean of its
windows. Empty texts get a zero vector.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


def load_biobert(model_name=MODEL_NAME, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # The model is cased; set it explicitly so a missing tokenizer config cannot lowercase input.
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()
    return tokenizer, model, device


def chunk_texts(texts, tokenizer, max_length=512, stride=64, max_chunks=None):
    """Tokenize texts into windows. Returns (windows, owner index per window)."""
    windows, owners = [], []
    for i, text in enumerate(texts):
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_length, stride=stride,
                        return_overflowing_tokens=True)
        ids = enc["input_ids"][:max_chunks] if max_chunks else enc["input_ids"]
        windows.extend(ids)
        owners.extend([i] * len(ids))
    return windows, np.array(owners, dtype=np.int64)


@torch.no_grad()
def embed_windows(windows, model, device, pad_id, batch_size=32):
    """Mean-pooled embedding and token count for each window."""
    hidden = model.config.hidden_size
    vecs = np.zeros((len(windows), hidden), dtype=np.float32)
    counts = np.array([len(w) for w in windows], dtype=np.float32)
    order = np.argsort(counts)  # batch similar lengths together to minimize padding
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = int(counts[idx].max())
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        for row, j in enumerate(idx):
            ids[row, :len(windows[j])] = torch.tensor(windows[j])
        ids = ids.to(device)
        mask = (ids != pad_id).unsqueeze(-1)
        out = model(input_ids=ids, attention_mask=mask.squeeze(-1).long()).last_hidden_state
        pooled = (out * mask).sum(1) / mask.sum(1)
        vecs[idx] = pooled.float().cpu().numpy()
    return vecs, counts


def embed_texts(texts, tokenizer, model, device, max_length=512, stride=64, max_chunks=None, batch_size=32):
    """Return an (n_texts, hidden) float32 array and the number of windows per text."""
    windows, owners = chunk_texts(texts, tokenizer, max_length, stride, max_chunks)
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    n_windows = np.bincount(owners, minlength=len(texts)) if len(owners) else np.zeros(len(texts), dtype=np.int64)
    if not windows:
        return out, n_windows
    vecs, counts = embed_windows(windows, model, device, tokenizer.pad_token_id, batch_size)
    np.add.at(out, owners, vecs * counts[:, None])
    weights = np.bincount(owners, weights=counts, minlength=len(texts))
    nonzero = weights > 0
    out[nonzero] /= weights[nonzero, None]
    return out, n_windows
