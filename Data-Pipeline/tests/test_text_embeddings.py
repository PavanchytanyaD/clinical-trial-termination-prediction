"""text and embeddings stages. BioBERT itself is replaced by a stub, so these
tests need neither torch nor a GPU nor Google Cloud."""

import json
import sys
import types

import numpy as np
import pytest

from scripts import embeddings, text

FIELDS = embeddings.FIELDS


@pytest.fixture()
def stub_biobert(monkeypatch):
    """src.features.biobert_embeddings with an 8-dim fake model."""
    def embed_texts(texts, tokenizer, model, device, **kw):
        vecs = np.array([[len(t)] * 8 for t in texts], dtype=np.float32)
        return vecs, np.array([1 if t else 0 for t in texts])

    config = types.SimpleNamespace(hidden_size=8)
    module = types.ModuleType("src.features.biobert_embeddings")
    module.load_biobert = lambda name: (None, types.SimpleNamespace(
        config=config), "cpu")
    module.embed_texts = embed_texts
    monkeypatch.setitem(sys.modules, "src.features.biobert_embeddings",
                        module)


@pytest.fixture()
def text_file(snapshot_dir, tmp_path, monkeypatch):
    """Run the text stage on the sample snapshot into tmp_path."""
    monkeypatch.setattr(text, "RAW_DIR", snapshot_dir)
    monkeypatch.setattr(text, "TEXT_JSONL", tmp_path / "text_all.jsonl")
    monkeypatch.setattr(text, "TEXT_META", tmp_path / "text_all.meta.json")
    monkeypatch.setattr(text, "tracked_step", fake_tracked_step)
    text.main()
    return tmp_path / "text_all.jsonl"


class fake_tracked_step:
    def __init__(self, name):
        self.run = {"run_id": "manual__2026-10-25T10:00:00+00:00",
                    "metrics": {}, "outputs": {}}

    def __enter__(self):
        return self.run

    def __exit__(self, *exc):
        return False


def test_text_stage_writes_every_trial(text_file):
    rows = [json.loads(line) for line in text_file.read_text().splitlines()]
    assert len(rows) == 300  # all trials, not only finished ones
    assert set(FIELDS) <= set(rows[0])
    meta = json.loads((text_file.parent / "text_all.meta.json").read_text())
    assert meta["all_trials"] is True and meta["n_rows"] == 300


def test_local_backend(text_file, tmp_path, monkeypatch, stub_biobert):
    out = tmp_path / "embeddings"
    monkeypatch.setattr(embeddings, "TEXT_JSONL", text_file)
    monkeypatch.setattr(embeddings, "EMB_DIR", out)
    monkeypatch.setattr(embeddings, "tracked_step", fake_tracked_step)
    monkeypatch.setattr(embeddings, "load_params", lambda: {
        "embeddings": {"backend": "local", "max_trials": 50}})
    embeddings.stage()
    ids = out.joinpath("ids.txt").read_text().split()
    assert len(ids) == 50
    for field in FIELDS:
        arr = np.load(out / f"{field}.npy")
        assert arr.shape == (50, 8) and arr.dtype == np.float16
    meta = json.loads((out / "meta.json").read_text())
    assert meta["backend"] == "local" and meta["rows"] == 50
    assert meta["fields"] == FIELDS


def test_unknown_backend(monkeypatch):
    monkeypatch.setattr(embeddings, "tracked_step", fake_tracked_step)
    monkeypatch.setattr(embeddings, "load_params",
                        lambda: {"embeddings": {"backend": "tpu"}})
    with pytest.raises(ValueError, match="vertex or local"):
        embeddings.stage()


def test_vertex_job_spec_uses_gpu_and_pipeline_sa():
    p = {**embeddings.DEFAULTS, "max_trials": 10}
    spec = embeddings.vertex_job_spec(p, "gs://b/in.jsonl", "gs://b/out",
                                      "run-1")
    pool = spec["jobSpec"]["workerPoolSpecs"][0]
    assert pool["machineSpec"]["acceleratorType"] == "NVIDIA_TESLA_T4"
    assert pool["machineSpec"]["acceleratorCount"] == 1
    assert spec["jobSpec"]["serviceAccount"].startswith("pipeline-sa@")
    args = pool["containerSpec"]["args"]
    settings = json.loads(args[args.index("--settings") + 1])
    assert settings == {k: p[k] for k in embeddings.EMBED_KEYS}
    assert args[args.index("--input") + 1] == "gs://b/in.jsonl"


class FakeSession:
    def __init__(self, states, fail_after=None):
        self.states, self.calls, self.fail_after = list(states), [], fail_after

    def get(self, url):
        self.calls.append(("get", url))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise KeyboardInterrupt
        return types.SimpleNamespace(json=lambda: {"state":
                                                   self.states.pop(0)})

    def post(self, url, json=None):
        self.calls.append(("post", url))
        return types.SimpleNamespace(
            status_code=200, json=lambda: {"name": "projects/p/jobs/1"})


def test_wait_for_job_returns_final_state():
    session = FakeSession(["JOB_STATE_PENDING", "JOB_STATE_RUNNING",
                           "JOB_STATE_SUCCEEDED"])
    state = embeddings.wait_for_job(session, "https://x/job", poll_seconds=0)
    assert state["state"] == "JOB_STATE_SUCCEEDED"
    assert ("post", "https://x/job:cancel") not in session.calls


def test_wait_for_job_cancels_when_stopped():
    session = FakeSession(["JOB_STATE_RUNNING"] * 5, fail_after=2)
    with pytest.raises(KeyboardInterrupt):
        embeddings.wait_for_job(session, "https://x/job", poll_seconds=0)
    assert session.calls[-1] == ("post", "https://x/job:cancel")


def test_vertex_backend_round_trip(text_file, tmp_path, monkeypatch,
                                   stub_biobert):
    """stage(): upload text, run the job (here: the worker, in process),
    download the result into EMB_DIR."""
    bucket = {}
    monkeypatch.setattr(embeddings, "gcs_upload",
                        lambda path, uri: bucket.__setitem__(
                            uri, open(path, "rb").read()))

    def gcs_download(uri, path):
        open(path, "wb").write(bucket[uri])

    def gcs_download_dir(uri, local_dir):
        local_dir.mkdir(parents=True, exist_ok=True)
        for key, data in bucket.items():
            if key.startswith(uri + "/"):
                (local_dir / key[len(uri) + 1:]).write_bytes(data)

    monkeypatch.setattr(embeddings, "gcs_download", gcs_download)
    monkeypatch.setattr(embeddings, "gcs_download_dir", gcs_download_dir)

    submitted = {}

    class Session(FakeSession):
        def post(self, url, json=None):
            submitted.update(json)
            args = json["jobSpec"]["workerPoolSpecs"][0]["containerSpec"][
                "args"]
            embeddings.worker(args[2], args[4], __import__("json").loads(
                args[6]))
            return super().post(url, json)

    monkeypatch.setattr(embeddings, "_session",
                        lambda project: Session(["JOB_STATE_SUCCEEDED"]))
    out = tmp_path / "embeddings"
    monkeypatch.setattr(embeddings, "TEXT_JSONL", text_file)
    monkeypatch.setattr(embeddings, "EMB_DIR", out)
    monkeypatch.setattr(embeddings, "tracked_step", fake_tracked_step)
    monkeypatch.setattr(embeddings, "load_params", lambda: {
        "embeddings": {"backend": "vertex", "max_trials": 20}})
    embeddings.stage()

    assert submitted["displayName"] == \
        "ctt-biobert-manual-2026-10-25t10-00-00-00-00"
    meta = json.loads((out / "meta.json").read_text())
    assert meta["backend"] == "vertex" and meta["rows"] == 20
    assert meta["vertex_job"] == "projects/p/jobs/1"
    assert np.load(out / "title.npy").shape == (20, 8)
