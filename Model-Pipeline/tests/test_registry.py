"""Promotion and rollback in the register stage, with a fake Vertex AI
Model Registry (no cloud access)."""

import json
import shutil
from types import SimpleNamespace

import pytest

from model_pipeline import registry
from model_pipeline.config import Context, load_params
from tests.helpers import Constant, fit_logreg, make_rows, write_artifact

PARENT = "projects/p/locations/us-central1/models/123"


class FakeHub:
    """Versions of one model: id -> aliases, artifact uri and labels."""

    def __init__(self):
        self.versions = {}
        self.uploads = []

    def add(self, uri, labels=None, aliases=()):
        vid = str(len(self.versions) + 1)
        self.versions[vid] = {"aliases": set(aliases), "uri": uri,
                              "labels": labels or {}}
        return vid

    def find(self, ref):
        for vid, v in self.versions.items():
            if ref == vid or ref in v["aliases"]:
                return vid
        raise LookupError(ref)

    def aliases(self, alias):
        return [vid for vid, v in self.versions.items()
                if alias in v["aliases"]]


def fake_aiplatform(hub):
    class Model:
        def __init__(self, name):
            self.version_id = hub.find(name.split("@")[1])
            self.resource_name = PARENT
            v = hub.versions[self.version_id]
            self.uri, self.labels = v["uri"], v["labels"]

        @staticmethod
        def list(filter):
            return [SimpleNamespace(resource_name=PARENT)] \
                if hub.versions else []

        @staticmethod
        def upload(**kw):
            hub.uploads.append(kw)
            first = not hub.versions
            vid = hub.add(kw["artifact_uri"], kw["labels"])
            if first or kw["is_default_version"]:
                for v in hub.versions.values():
                    v["aliases"].discard("default")
                hub.versions[vid]["aliases"].add("default")
            return Model(f"{PARENT}@{vid}")

    class ModelRegistry:
        def __init__(self, name):
            pass

        def list_versions(self):
            return [SimpleNamespace(version_id=vid,
                                    version_aliases=sorted(v["aliases"]),
                                    version_create_time="t")
                    for vid, v in hub.versions.items()]

        def add_version_aliases(self, aliases, version):
            for a in aliases:
                # mirror Vertex AI: one version per alias, one default
                assert a == "default" or not hub.aliases(a), \
                    f"alias {a} still on {hub.aliases(a)}"
                if a == "default":
                    for v in hub.versions.values():
                        v["aliases"].discard("default")
                hub.versions[version]["aliases"].add(a)

        def remove_version_aliases(self, aliases, version):
            for a in aliases:
                hub.versions[version]["aliases"].discard(a)

    return SimpleNamespace(Model=Model,
                           models=SimpleNamespace(ModelRegistry=ModelRegistry))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    hub = FakeHub()
    bucket = tmp_path / "bucket"
    alerts = []
    monkeypatch.setattr(registry, "_aiplatform",
                        lambda reg: fake_aiplatform(hub))

    def upload(ctx, reg, run_id):
        dest = bucket / run_id
        shutil.copytree(ctx.path("artifact"), dest)
        return str(dest)

    monkeypatch.setattr(registry, "upload_artifact", upload)
    monkeypatch.setattr(registry, "download_artifact",
                        lambda uri, dest: shutil.copytree(uri, dest))
    monkeypatch.setattr(registry.notify, "alert",
                        lambda ctx, subject, body: alerts.append(subject))
    for name in ("set_tags", "log_metrics"):
        monkeypatch.setattr(registry.tracking, name, lambda *a, **k: None)
    return SimpleNamespace(hub=hub, bucket=bucket, alerts=alerts,
                           tmp=tmp_path)


def work_folder(tmp, name, model, test_rows):
    ctx = Context(tmp / name, load_params())
    ctx.update_state(run_id=name)
    write_artifact(ctx.path("artifact"), model)
    test_rows.to_pickle(ctx.path("data", "test.pkl"))
    return ctx


def champion(hub, folder, stored_pr_auc):
    """A registered champion whose stored label came from its own (older,
    different) test set."""
    vid = hub.add(str(folder),
                  {"test_pr_auc_x1000": str(int(stored_pr_auc * 1000))},
                  aliases={"champion", "default"})
    return vid


def test_first_model_becomes_champion(setup):
    rows = make_rows()
    ctx = work_folder(setup.tmp, "run-1", fit_logreg(rows), rows)
    out = registry.run(ctx)
    assert out["promoted"] and out["version"] == "1"
    assert setup.hub.versions["1"]["aliases"] == \
        {"champion", "challenger", "default"}
    upload = setup.hub.uploads[0]
    assert upload["serving_container_predict_route"] == "/predict"
    assert upload["serving_container_image_uri"].endswith(
        "model-serving:latest")


def test_champion_is_rescored_on_the_new_test_set(setup):
    """The old champion's stored PR-AUC (0.95, its own test set) is higher
    than the new model's, but on the new test rows it is a constant model
    and much worse: the new model is promoted."""
    rows = make_rows()
    old = write_artifact(setup.tmp / "old", Constant(0.4))
    champion(setup.hub, old, 0.95)
    ctx = work_folder(setup.tmp, "run-2", fit_logreg(rows), rows)
    out = registry.run(ctx)

    assert out["promoted"]
    assert out["champion_test_pr_auc_same_rows"] == \
        pytest.approx(rows["label"].mean())        # constant scores
    assert out["new_test_pr_auc"] > out["champion_test_pr_auc_same_rows"]
    v = setup.hub.versions
    assert v["2"]["aliases"] == {"champion", "challenger", "default"}
    assert v["1"]["aliases"] == set()
    assert v["2"]["labels"]["champion_version"] == "1"
    assert v["2"]["labels"]["champion_pr_auc_x1000"] == \
        str(int(round(out["champion_test_pr_auc_same_rows"] * 1000)))
    saved = json.loads(ctx.path("reports", "registry.json").read_text())
    assert saved["champion_version_before"] == "1"


def test_worse_model_is_not_promoted(setup):
    rows = make_rows()
    good = write_artifact(setup.tmp / "old", fit_logreg(rows))
    champion(setup.hub, good, 0.10)           # low stored label, good model
    ctx = work_folder(setup.tmp, "run-2", Constant(0.4), rows)
    out = registry.run(ctx)

    assert not out["promoted"]
    v = setup.hub.versions
    assert v["1"]["aliases"] == {"champion", "default"}
    assert v["2"]["aliases"] == {"challenger"}
    assert setup.hub.uploads[0]["is_default_version"] is False
    assert any("NOT promoted" in a for a in setup.alerts)


def test_min_improvement_is_applied(setup):
    rows = make_rows()
    model = fit_logreg(rows)
    champion(setup.hub, write_artifact(setup.tmp / "old", model), 0.5)
    ctx = work_folder(setup.tmp, "run-2", model, rows)    # same model
    assert registry.run(ctx)["promoted"]                   # tie: not worse
    ctx.params["gate"]["min_improvement"] = 0.01
    ctx2 = work_folder(setup.tmp, "run-3", model, rows)
    ctx2.params["gate"]["min_improvement"] = 0.01
    assert not registry.run(ctx2)["promoted"]


def test_unscorable_champion_is_kept_by_default(setup):
    rows = make_rows()
    other = rows.rename(columns={"f2": "g2"})
    old = write_artifact(setup.tmp / "old", fit_logreg(other,
                                                       ["f0", "f1", "g2"]),
                         columns=["f0", "f1", "g2"])
    champion(setup.hub, old, 0.5)
    ctx = work_folder(setup.tmp, "run-2", fit_logreg(rows), rows)
    out = registry.run(ctx)
    assert not out["promoted"] and "features" in out["reason"]


def test_rollback_moves_champion_and_default(setup):
    hub = setup.hub
    hub.add("a")
    hub.add("b", aliases={"champion", "challenger", "default"})
    before = registry.rollback(load_params()["registry"], "1")
    assert before == "2"
    assert hub.versions["1"]["aliases"] == {"champion", "default"}
    assert hub.versions["2"]["aliases"] == {"challenger"}
    with pytest.raises(SystemExit):
        registry.rollback(load_params()["registry"], "9")


def test_rollback_command_sends_an_alert(setup):
    setup.hub.add("a", aliases={"default", "champion"})
    setup.hub.add("b")
    registry.main(["rollback", "--version", "2"])
    assert setup.hub.versions["2"]["aliases"] == {"champion", "default"}
    assert setup.alerts == ["manual rollback to version 2"]
