"""Settings and the work folder shared by all stages of one run."""

import json
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).resolve().parents[1]   # Model-Pipeline/
PARAMS = PIPELINE_DIR / "params.yaml"


def load_params(path=PARAMS):
    return yaml.safe_load(Path(path).read_text())


class Context:
    """One run of the pipeline: settings, the local work folder and a small
    state file that later stages read (run id, snapshot, MLflow run)."""

    def __init__(self, work, params, snapshot=None, source=None):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.params = params
        if snapshot:
            self.params["data"]["snapshot"] = snapshot
        if source:
            self.params["data"]["source"] = source

    def path(self, *parts):
        p = self.work.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read_json(self, name):
        return json.loads(self.path(name).read_text())

    def write_json(self, name, obj):
        self.path(name).write_text(json.dumps(obj, indent=2, default=_plain))

    @property
    def state(self):
        p = self.path("state.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def update_state(self, **values):
        self.write_json("state.json", {**self.state, **values})


def _plain(value):
    """JSON for numpy numbers and paths."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)
