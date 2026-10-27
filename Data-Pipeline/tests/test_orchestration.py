"""Stage runner, run tracking, alerts and the Airflow DAG."""

import json

import pytest
import yaml

from scripts import alerts, run_stage, tracking
from scripts.config import ROOT


# ---------------------------------------------------------------- run_stage
def test_param_override_keeps_comments_and_types(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("# top comment\nacquire:\n  source: sample  # where\n"
                 "  snapshot: \"20261001\"\n  sample_trials: 500\n"
                 "validate:\n  fail_on: critical\n")
    run_stage.set_params(["acquire.snapshot=20261101",
                          "acquire.sample_trials=900",
                          "acquire.source=zip", "acquire.zip=gs://b/x.zip",
                          "validate.alert_on=never"], p)
    text = p.read_text()
    assert "# top comment" in text and "# where" in text
    params = yaml.safe_load(text)
    assert params["acquire"] == {"source": "zip", "snapshot": "20261101",
                                 "sample_trials": 900, "zip": "gs://b/x.zip"}
    assert params["validate"] == {"fail_on": "critical", "alert_on": "never"}


def test_empty_number_override_keeps_params_value(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("acquire:\n  sample_trials: 500\n  zip: \"x\"\n")
    run_stage.set_params(["acquire.sample_trials=", "acquire.zip="], p)
    assert yaml.safe_load(p.read_text())["acquire"] == {"sample_trials": 500,
                                                        "zip": ""}


@pytest.mark.parametrize("bad", ["snapshot=1", "acquire=1", "a.b.c=1", "x"])
def test_bad_overrides_are_rejected(bad, tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("acquire:\n  snapshot: \"1\"\n")
    with pytest.raises(ValueError):
        run_stage.set_params([bad], p)


def test_stage_graph_matches_dvc_yaml():
    assert list(run_stage.stages()) == ["acquire", "preprocess", "bias",
                                        "bias_model", "schema_stats",
                                        "anomalies", "validate", "text",
                                        "embeddings"]
    produced = run_stage.produced_by()
    assert produced["data/raw/aact_snapshot"] == "acquire"
    assert produced["data/processed/structured_features.csv"] == "preprocess"
    # metrics with cache: false are not pulled from the remote
    assert "reports/validation/summary.json" not in produced
    assert {produced.get(d) for d in run_stage.stage_deps("bias_model")} \
        >= {"preprocess", "bias"}
    upstream = {produced.get(d) for d in run_stage.stage_deps("validate")}
    assert upstream >= {"preprocess", "schema_stats"}
    assert produced["data/processed/embeddings"] == "embeddings"
    assert {produced.get(d) for d in run_stage.stage_deps("embeddings")} \
        >= {"text"}


def test_image_copies_every_folder_the_stages_read():
    """Each pod runs the pipeline image, so every ../ dependency in
    dvc.yaml (team code, shared configs) must be copied into it."""
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    copied = {line.split()[1].rstrip("/") for line in dockerfile.splitlines()
              if line.startswith("COPY ")}
    needed = {dep.split("/")[1] for stage in run_stage.stages()
              for dep in run_stage.stage_deps(stage) if dep.startswith("../")}
    assert needed <= copied, f"not in the image: {sorted(needed - copied)}"


def test_unknown_stage(monkeypatch):
    monkeypatch.delenv("PIPELINE_STATE_URI", raising=False)
    with pytest.raises(ValueError, match="unknown stage"):
        run_stage.run_stage("train")


def test_gcs_uri_split():
    with pytest.raises(ValueError):
        run_stage._split("s3://bucket/x")
    assert run_stage._split("gs://b/data-pipeline/state") == \
        ("b", "data-pipeline/state")


def test_local_state_folder_round_trip(tmp_path):
    state = tmp_path / "state"
    src = tmp_path / "a.txt"
    src.write_text("hello")
    assert run_stage.download(f"{state}/a.txt", tmp_path / "b.txt") is False
    run_stage.upload(src, f"{state}/runs/r1/a.txt")
    assert run_stage.download(f"{state}/runs/r1/a.txt", tmp_path / "c.txt")
    assert (tmp_path / "c.txt").read_text() == "hello"
    folder = tmp_path / "reports"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "x.md").write_text("x")
    run_stage.upload_tree(folder, f"{state}/baseline")
    assert run_stage.download_tree(f"{state}/baseline", tmp_path / "copy")
    assert (tmp_path / "copy" / "sub" / "x.md").read_text() == "x"
    assert run_stage.download_tree(f"{state}/none", tmp_path / "n") is False


def test_local_write_is_conditional(tmp_path):
    uri = str(tmp_path / "state" / "dvc.lock")
    assert run_stage.read_versioned(uri) == (None, 0)
    assert run_stage.write_if_generation(uri, "one", 0)
    assert run_stage.write_if_generation(uri, "again", 0) is False
    text, generation = run_stage.read_versioned(uri)
    assert text == "one"
    assert run_stage.write_if_generation(uri, "two", generation)
    assert run_stage.write_if_generation(uri, "three", generation) is False
    assert run_stage.read_versioned(uri)[0] == "two"


def test_push_lock_to_a_local_state_folder(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "dvc.lock").write_text(
        yaml.safe_dump({"stages": {"acquire": {"cmd": "a"}}}))
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.safe_dump({"stages": {"bias": {"cmd": "b"}}}))
    monkeypatch.setattr(run_stage, "LOCK_FILE", lock)
    run_stage.State(str(state), "r1").push_lock("bias")
    shared = yaml.safe_load((state / "dvc.lock").read_text())
    assert set(shared["stages"]) == {"acquire", "bias"}
    assert (state / "runs" / "r1" / "dvc.lock").exists()


def test_dvc_remote_override(monkeypatch):
    calls = []
    monkeypatch.setattr(run_stage, "dvc", lambda *a, **k: calls.append(a))
    monkeypatch.delenv("PIPELINE_DVC_REMOTE", raising=False)
    monkeypatch.delenv("PIPELINE_GCP_PROJECT", raising=False)
    run_stage.configure_remote()
    assert calls == []
    monkeypatch.setenv("PIPELINE_DVC_REMOTE", "/opt/ctt-output/dvc-store")
    run_stage.configure_remote()
    assert calls == [("remote", "add", "--local", "--force", "--default",
                      "pipeline", "/opt/ctt-output/dvc-store")]
    calls.clear()
    monkeypatch.setenv("PIPELINE_DVC_REMOTE", "gs://their-bucket/dvc-store")
    monkeypatch.setenv("PIPELINE_GCP_PROJECT", "their-project")
    run_stage.configure_remote()
    assert calls[1] == ("remote", "modify", "--local", "pipeline",
                        "projectname", "their-project")


# ---------------------------------------------------------------- tracking
@pytest.fixture()
def run_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path / "logs" / "runs")
    monkeypatch.setenv("PIPELINE_RUN_ID", "test_run")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path / "logs"


def records(log_dir):
    lines = (log_dir / "runs" / "test_run.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_tracked_step_records_success(run_dirs, tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "setup_logging",
                        lambda name: tracking.logging.getLogger(name))
    out = tmp_path / "out.txt"
    out.write_text("hello")
    with tracking.tracked_step("demo") as run:
        run["metrics"]["rows"] = 3
        run["outputs"]["out.txt"] = tracking.fingerprint(out)
    (rec,) = records(run_dirs)
    assert rec["status"] == "success" and rec["metrics"] == {"rows": 3}
    md5_of_hello = "5d41402abc4b2a76b9719d911017c592"
    assert rec["outputs"]["out.txt"]["md5"] == md5_of_hello
    assert rec["duration_seconds"] >= 0


def test_tracked_step_records_failure_and_alerts(run_dirs, monkeypatch):
    monkeypatch.setattr(tracking, "setup_logging",
                        lambda name: tracking.logging.getLogger(name))
    monkeypatch.delenv("PIPELINE_ORCHESTRATED", raising=False)
    sent = []
    monkeypatch.setattr(tracking, "send_alert",
                        lambda subject, body: sent.append(subject))
    with pytest.raises(RuntimeError):
        with tracking.tracked_step("demo"):
            raise RuntimeError("boom")
    (rec,) = records(run_dirs)
    assert rec["status"] == "failed" and "boom" in rec["error"]
    assert sent == ["step demo failed"]


@pytest.mark.parametrize("attempt, tries, alerted",
                         [("1", "3", False), ("3", "3", True), ("", "", True)])
def test_task_failure_alerts_only_on_the_last_attempt(
        attempt, tries, alerted, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_TRY", attempt)
    monkeypatch.setenv("PIPELINE_MAX_TRIES", tries)
    monkeypatch.setattr(run_stage, "LOG_DIR", tmp_path)
    (tmp_path / "bias.log").write_text("line 1\nKeyError: 'phase'\n")
    sent = []
    monkeypatch.setattr(run_stage, "send_alert",
                        lambda subject, body: sent.append((subject, body)))
    run_stage.alert_failure("bias", "RuntimeError: stage bias failed")
    assert bool(sent) == alerted
    if sent:
        subject, body = sent[0]
        assert subject == "task bias failed"
        assert "KeyError: 'phase'" in body and "stage bias failed" in body


def test_no_step_alert_when_airflow_alerts(run_dirs, monkeypatch):
    monkeypatch.setattr(tracking, "setup_logging",
                        lambda name: tracking.logging.getLogger(name))
    monkeypatch.setenv("PIPELINE_ORCHESTRATED", "1")
    sent = []
    monkeypatch.setattr(tracking, "send_alert",
                        lambda subject, body: sent.append(subject))
    with pytest.raises(RuntimeError):
        with tracking.tracked_step("demo"):
            raise RuntimeError("boom")
    assert sent == []


def test_fingerprint_of_a_folder(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    fp = tracking.fingerprint(tmp_path)
    assert fp["files"] == 2 and fp["bytes"] == 2
    (tmp_path / "b.txt").write_text("3")
    assert tracking.fingerprint(tmp_path)["md5"] != fp["md5"]


# ---------------------------------------------------------------- alerts
SETTINGS = {"host": "smtp.test", "port": 587, "user": "u", "password": "p",
            "starttls": True, "sender": "bot@test", "recipients": ["a@test",
                                                                   "b@test"]}


def test_alert_is_skipped_without_smtp():
    assert alerts.send_alert("x", "y", {**SETTINGS, "host": ""}) is False


def test_alert_is_sent(monkeypatch):
    calls = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            calls["tls"] = True

        def login(self, user, password):
            calls["login"] = user

        def send_message(self, msg):
            calls["msg"] = msg

    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    assert alerts.send_alert("validation critical", "details", SETTINGS)
    msg = calls["msg"]
    assert calls["server"] == ("smtp.test", 587) and calls["tls"]
    assert msg["Subject"] == "[ctt-data-pipeline] validation critical"
    assert msg["To"] == "a@test, b@test"
    assert "details" in msg.get_content()


def test_alert_failure_does_not_raise(monkeypatch):
    def broken(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(alerts.smtplib, "SMTP", broken)
    assert alerts.send_alert("x", "y", SETTINGS) is False


# ---------------------------------------------------------------- DAG
@pytest.fixture(autouse=True)
def composer_env(monkeypatch):
    """The DAG reads CTT_* when it is loaded: start from Composer's
    defaults."""
    for name in ["RUNNER", "STATE_URI", "DVC_REMOTE", "GCP_PROJECT",
                 "REGION", "BUCKET", "GPU_IMAGE", "SERVICE_ACCOUNT",
                 "EMBEDDINGS", "SCHEDULE"]:
        monkeypatch.delenv(f"CTT_{name}", raising=False)


def load_dag(dag_id="ctt_data_pipeline"):
    pytest.importorskip("airflow.models")
    pytest.importorskip("airflow.providers.cncf.kubernetes")
    from airflow.models import DagBag
    bag = DagBag(dag_folder=str(ROOT / "dags"), include_examples=False)
    assert bag.import_errors == {}
    return bag.get_dag(dag_id)


def test_dag_structure():
    dag = load_dag()
    upstream = {t.task_id: sorted(t.upstream_task_ids) for t in dag.tasks}
    assert upstream == {
        "acquire": [],
        "preprocess": ["acquire"],
        "bias": ["acquire"],
        "text": ["acquire"],
        "embeddings": ["text"],
        "bias_model": ["bias", "preprocess"],
        "schema_stats": ["preprocess"],
        "anomalies": ["preprocess"],
        "validate": ["schema_stats"],
        "publish": ["anomalies", "bias_model", "embeddings", "validate"],
    }
    for t in dag.tasks:
        assert type(t).__name__ == "KubernetesPodOperator"
        assert t.retries == 2
    assert dag.max_active_runs == 1 and dag.catchup is False


def test_schedule_defaults_to_monthly_and_can_be_changed(monkeypatch):
    assert load_dag().schedule_interval == "0 6 5 * *"
    monkeypatch.setenv("CTT_SCHEDULE", "@weekly")
    assert load_dag().schedule_interval == "@weekly"
    from airflow.models import Variable
    Variable.set("ctt_schedule", "0 6 * * 1")   # the Variable wins
    try:
        assert load_dag().schedule_interval == "0 6 * * 1"
        Variable.set("ctt_schedule", "none")
        assert load_dag().schedule_interval is None
    finally:
        Variable.delete("ctt_schedule")


def test_freeze_dag_runs_one_pod():
    dag = load_dag("ctt_freeze_baseline")
    (task,) = dag.tasks
    assert task.arguments[:2] == ["freeze", "--reason"]
    assert dag.schedule_interval is None


def test_tasks_pass_params_as_separate_args():
    dag = load_dag()
    args = dag.get_task("acquire").arguments
    assert args[0] == "acquire" and args.count("--param") == 4
    assert not any("'" in a for a in args)  # no shell quoting reaches the pod
    env = {e.name: e.value for e in dag.get_task("validate").env_vars}
    assert env["PIPELINE_RUN_ID"] == "{{ run_id }}"
    assert env["PIPELINE_ORCHESTRATED"] == "1"
    assert env["PIPELINE_STATE_URI"].startswith("gs://")


def test_deployment_settings_reach_every_pod(monkeypatch):
    monkeypatch.setenv("CTT_GCP_PROJECT", "their-project")
    monkeypatch.setenv("CTT_DVC_REMOTE", "gs://their-project-data/dvc-store")
    dag = load_dag()
    env = {e.name: e.value for e in dag.get_task("embeddings").env_vars}
    assert env["PIPELINE_GCP_PROJECT"] == "their-project"
    assert env["PIPELINE_DVC_REMOTE"] == "gs://their-project-data/dvc-store"
    assert "PIPELINE_BUCKET" not in env  # unset: the stage's default


def test_local_runner_runs_each_stage_in_bash(monkeypatch):
    monkeypatch.setenv("CTT_RUNNER", "local")
    monkeypatch.setenv("CTT_STATE_URI", "/opt/ctt-output/state")
    monkeypatch.setenv("CTT_EMBEDDINGS", "local")
    dag = load_dag()
    assert len(dag.tasks) == 10
    for t in dag.tasks:
        assert type(t).__name__ == "BashOperator"
        assert "local/run_task.sh" in t.bash_command
        assert t.env["PIPELINE_STATE_URI"] == "/opt/ctt-output/state"
        assert t.env["PIPELINE_EMBEDDINGS"] == "local"
        assert t.append_env is True
    cmd = dag.get_task("acquire").bash_command
    assert cmd.count("--param") == 4
    # each override is one shell word, its template intact
    assert "'acquire.snapshot={{ dag_run.conf.get(\"snapshot\")" in cmd
    (freeze,) = load_dag("ctt_freeze_baseline").tasks
    assert "freeze --reason" in freeze.bash_command


# ---------------------------------------------------------------- dvc.lock
def test_merge_lock_entry_keeps_other_stages():
    shared = yaml.safe_dump({"schema": "2.0", "stages": {
        "acquire": {"cmd": "a"}, "bias": {"cmd": "old"}}})
    own = {"schema": "2.0", "stages": {"bias": {"cmd": "new"},
                                       "acquire": {"cmd": "stale"}}}
    merged = yaml.safe_load(run_stage.merge_lock_entry(shared, own, "bias"))
    assert merged["stages"] == {"acquire": {"cmd": "a"},
                                "bias": {"cmd": "new"}}


def test_push_lock_retries_when_another_stage_wrote_first(tmp_path,
                                                          monkeypatch):
    store = {"text": yaml.safe_dump({"stages": {"acquire": {"cmd": "a"}}}),
             "gen": 1}
    writes = []

    def read_versioned(uri):
        return store["text"], store["gen"]

    def write_if_generation(uri, text, generation):
        if not writes:  # a parallel stage commits between our read and write
            writes.append("other")
            other = yaml.safe_load(store["text"])
            other["stages"]["preprocess"] = {"cmd": "p"}
            store.update(text=yaml.safe_dump(other), gen=store["gen"] + 1)
        if generation != store["gen"]:
            return False
        store.update(text=text, gen=generation + 1)
        writes.append("ours")
        return True

    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.safe_dump({"stages": {"bias": {"cmd": "b"}}}))
    monkeypatch.setattr(run_stage, "LOCK_FILE", lock)
    monkeypatch.setattr(run_stage, "read_versioned", read_versioned)
    monkeypatch.setattr(run_stage, "write_if_generation",
                        write_if_generation)
    monkeypatch.setattr(run_stage, "upload", lambda path, uri: None)
    monkeypatch.setattr(run_stage.time, "sleep", lambda s: None)
    run_stage.State("gs://b/state", "r1").push_lock("bias")
    stages = yaml.safe_load(store["text"])["stages"]
    assert set(stages) == {"acquire", "preprocess", "bias"}
    assert writes == ["other", "ours"]
    assert set(yaml.safe_load(lock.read_text())["stages"]) == set(stages)
