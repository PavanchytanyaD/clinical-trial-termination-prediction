"""A new `--stage all` starts a new run; a single stage continues it."""

import json

from model_pipeline import run as runner
from model_pipeline import tracking
from model_pipeline.config import load_params


def test_stage_all_starts_fresh_and_single_stage_continues(tmp_path,
                                                           monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    params = load_params()
    monkeypatch.setattr(runner, "load_params", lambda: params)
    monkeypatch.setattr("model_pipeline.notify.alert", lambda *a, **k: None)
    # Each stage only logs one param, with a value that differs per call.
    calls = iter(range(100))
    monkeypatch.setattr(runner, "run_stage", lambda ctx, name:
                        tracking.log_params(ctx, {"stage_param": next(calls)}))
    work = tmp_path / "work"

    def state():
        return json.loads((work / "state.json").read_text())

    runner.main(["--stage", "all", "--workdir", str(work),
                 "--skip-register"])
    first = state()
    (work / "reports").mkdir(exist_ok=True)
    (work / "reports" / "old.txt").write_text("from the first run")

    runner.main(["--stage", "validate", "--workdir", str(work)])
    assert state() == first
    assert (work / "reports" / "old.txt").exists()

    runner.main(["--stage", "all", "--workdir", str(work),
                 "--skip-register"])
    second = state()
    assert second["run_id"] != first["run_id"]
    assert second["mlflow_run_id"] != first["mlflow_run_id"]
    assert not (work / "reports" / "old.txt").exists()
    assert (work / "mlruns").exists()           # earlier runs are kept
