"""Tests for the embedding code using a tiny randomly initialized BERT.

No model download is needed: the tokenizer and model are built in the test.
"""

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.features import biobert_embeddings
from src.features.biobert_embeddings import chunk_texts, embed_texts

WORDS = [f"w{i}" for i in range(50)]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab.write_text("\n".join(special + WORDS), encoding="utf-8")
    return BertTokenizerFast(str(vocab), do_lower_case=False)


@pytest.fixture(scope="module")
def model(tokenizer):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                        num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64)
    return BertModel(config).eval()


def words(n):
    return " ".join(WORDS[i % len(WORDS)] for i in range(n))


def direct_mean(model, tokenizer, text):
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return model(input_ids=ids).last_hidden_state.mean(1)[0].numpy()


def test_chunk_texts_windows_and_overlap(tokenizer):
    windows, owners = chunk_texts(["", words(5), words(60)], tokenizer,
                                  max_length=16, stride=4)
    assert owners.tolist()[0] == 1  # empty text has no windows
    long_windows = [w for w, o in zip(windows, owners) if o == 2]
    assert len(long_windows) > 1
    assert all(len(w) <= 16 for w in windows)
    # Consecutive windows repeat the last `stride` tokens (before [SEP]).
    assert long_windows[0][-5:-1] == long_windows[1][1:5]


def test_chunk_texts_max_chunks(tokenizer):
    _, owners = chunk_texts([words(200)], tokenizer, max_length=16,
                            stride=4, max_chunks=3)
    assert len(owners) == 3


def test_embed_texts_shapes_and_empty(tokenizer, model):
    texts = [words(3), "", words(60)]
    vecs, n_windows = embed_texts(texts, tokenizer, model, "cpu",
                                  max_length=16, stride=4)
    assert vecs.shape == (3, 16) and vecs.dtype == np.float32
    assert not vecs[1].any() and n_windows[1] == 0
    assert n_windows[0] == 1 and n_windows[2] > 1
    assert not np.isnan(vecs).any()


def test_short_text_matches_direct_mean_pooling(tokenizer, model):
    vecs, _ = embed_texts([words(5)], tokenizer, model, "cpu")
    expected = direct_mean(model, tokenizer, words(5))
    np.testing.assert_allclose(vecs[0], expected, atol=1e-5)


def test_long_text_is_token_weighted_mean_of_windows(tokenizer, model):
    text = words(40)
    windows, _ = chunk_texts([text], tokenizer, max_length=16, stride=4)
    pieces = [tokenizer.decode(w, skip_special_tokens=True) for w in windows]
    weights = np.array([len(w) for w in windows], dtype=np.float64)
    expected = sum(direct_mean(model, tokenizer, p) * n
                   for p, n in zip(pieces, weights)) / weights.sum()
    vecs, _ = embed_texts([text], tokenizer, model, "cpu", max_length=16,
                          stride=4)
    np.testing.assert_allclose(vecs[0], expected, atol=1e-5)


def test_batching_does_not_change_results(tokenizer, model):
    texts = [words(n) for n in (3, 9, 30, 2, 60)]
    a, _ = embed_texts(texts, tokenizer, model, "cpu", max_length=16,
                       stride=4, batch_size=1)
    b, _ = embed_texts(texts, tokenizer, model, "cpu", max_length=16,
                       stride=4, batch_size=4)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_out_of_memory_splits_batch(tokenizer, model, monkeypatch):
    real = biobert_embeddings._pool_batch
    sizes = []

    def fake(windows, idx, *args):
        sizes.append(len(idx))
        if len(idx) > 2:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return real(windows, idx, *args)

    monkeypatch.setattr(biobert_embeddings, "_pool_batch", fake)
    texts = [words(n) for n in (3, 5, 7, 9, 11)]
    vecs, _ = embed_texts(texts, tokenizer, model, "cpu", batch_size=8)
    monkeypatch.setattr(biobert_embeddings, "_pool_batch", real)
    expected, _ = embed_texts(texts, tokenizer, model, "cpu", batch_size=8)
    # 5 fails -> halves of 2 and 3; 3 fails -> halves of 1 and 2.
    assert sizes == [5, 2, 3, 1, 2]
    np.testing.assert_allclose(vecs, expected, atol=1e-5)
