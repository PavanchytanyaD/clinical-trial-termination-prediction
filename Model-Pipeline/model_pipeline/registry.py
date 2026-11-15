"""Stage register: push the model to the Vertex AI Model Registry, promote
it only if it is not worse than the current champion, and roll back by hand
if needed.

Every run that reaches this stage uploads a new version of
ctt-termination-model (served by our own container, serving/) and moves the
alias "challenger" to it. Before that, the current champion's artifact is
downloaded and scored on this run's test set, so both models are compared on
the same trials (each data version has a different test set, so the PR-AUC
stored on an old version is not comparable). The alias "champion" (the model
the inference service loads) and the registry's default version move to the
new version only if its test PR-AUC is at least the champion's on the same
rows plus min_improvement. Otherwise the champion stays and an alert says
so. Both scores go to reports/registry.json and the version's labels.

Manual rollback or a look at the versions:
    python -m model_pipeline.registry status
    python -m model_pipeline.registry rollback --version 3
"""

import argparse
import logging
import os
import tempfile

from model_pipeline import artifact, data, metrics, notify, tracking
from model_pipeline.config import Context, load_params

logger = logging.getLogger(__name__)
CHAMPION = "champion"
CHALLENGER = "challenger"
DEFAULT = "default"


def label(value):
    """Vertex AI label values: lowercase letters, digits, - and _."""
    text = str(value).lower()
    clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in text)
    return clean[:63] or "none"


def x1000(value):
    return "none" if value is None else label(int(round(value * 1000)))


def should_promote(new_score, champion_score, min_improvement):
    """True if there is no champion yet or the new model is not worse
    (both scores on the same test rows)."""
    if champion_score is None:
        return True
    return new_score >= champion_score + min_improvement


def _aiplatform(reg):
    from google.cloud import aiplatform
    aiplatform.init(project=reg["project"], location=reg["region"])
    return aiplatform


def registry_for(aiplatform, parent):
    return aiplatform.models.ModelRegistry(parent.resource_name)


def find_parent(aiplatform, name):
    models = aiplatform.Model.list(filter=f'display_name="{name}"')
    return models[0] if models else None


def get_champion(aiplatform, parent):
    if parent is None:
        return None
    try:
        return aiplatform.Model(f"{parent.resource_name}@{CHAMPION}")
    except Exception:
        return None


def holder_of(registry, alias):
    """Version id that has the alias now, or None."""
    for v in registry.list_versions():
        if alias in (v.version_aliases or []):
            return str(v.version_id)
    return None


def move_alias(registry, alias, version):
    """Give `alias` to `version`, taking it from the version that has it
    (an alias belongs to one version at a time). The default alias cannot
    be removed, only moved, so it is just added to the new version."""
    version = str(version)
    old = holder_of(registry, alias)
    if old == version:
        return
    if old is not None and alias != DEFAULT:
        registry.remove_version_aliases([alias], version=old)
    registry.add_version_aliases([alias], version=version)


def make_champion(registry, version):
    """Champion alias and default version follow each other."""
    move_alias(registry, CHAMPION, version)
    move_alias(registry, DEFAULT, version)


def upload_artifact(ctx, reg, run_id):
    import gcsfs
    bucket = ctx.params["data"]["bucket"]
    dest = f"{bucket}/{reg['artifacts_prefix']}/{run_id}"
    fs = gcsfs.GCSFileSystem()
    for f in ctx.path("artifact").iterdir():
        if f.is_file() and f.name != ".keep":
            fs.put_file(str(f), f"{dest}/{f.name}")
    return f"gs://{dest}"


def download_artifact(uri, dest):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    src = uri.removeprefix("gs://").rstrip("/")
    os.makedirs(dest, exist_ok=True)
    for path in fs.ls(src):
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if fs.isfile(path):
            fs.get_file(path, os.path.join(dest, name))
    return dest


def test_pr_auc(packaged, test):
    """PR-AUC of a packaged model on the test rows; None if it cannot score
    them (e.g. it needs features this data version does not have)."""
    missing = packaged.missing_columns(test.columns)
    if missing:
        logger.warning("%s needs %d features not in this test set: %s",
                       packaged.name, len(missing), missing[:5])
        return None
    s = packaged.score_frame(test)
    return metrics.compute(test["label"], s)["pr_auc"]


def score_champion(champion, test, work):
    """(PR-AUC of the champion on this test set or None, reason)."""
    try:
        folder = download_artifact(champion.uri, work)
        score = test_pr_auc(artifact.load(folder), test)
    except Exception as exc:
        logger.warning("champion could not be scored: %s", exc)
        return None, f"champion artifact not usable ({exc})"
    if score is None:
        return None, "champion needs features this data version lacks"
    return score, None


def decide(new, old, unscorable_reason, has_champion, reg, gate):
    """(promote?, reason)."""
    if not has_champion:
        return True, "first champion"
    if old is None:
        promote = reg.get("on_champion_unscorable", "keep") == "promote"
        return promote, unscorable_reason
    if should_promote(new, old, gate["min_improvement"]):
        return True, f"test PR-AUC {new:.4f} >= champion {old:.4f}"
    return False, f"test PR-AUC {new:.4f} < champion {old:.4f}"


def serving_image(reg):
    return os.environ.get("CTT_SERVING_IMAGE") or \
        reg["serving_container_image_uri"]


def run(ctx):
    reg, gate = ctx.params["registry"], ctx.params["gate"]
    card = ctx.read_json("artifact/model_card.json")
    cand = card["candidate"]
    run_id = ctx.state.get("run_id", "manual")
    test = data.load_split(ctx, "test")
    aiplatform = _aiplatform(reg)

    # both models on this run's test rows
    new = test_pr_auc(artifact.load(ctx.path("artifact")), test)
    parent = find_parent(aiplatform, reg["model_name"])
    champion = get_champion(aiplatform, parent)
    old, why = None, None
    if champion is not None:
        old, why = score_champion(champion, test,
                                  str(ctx.path("champion_artifact")))
    promoted, reason = decide(new, old, why, champion is not None, reg, gate)
    champion_version = champion.version_id if champion else None

    uri = upload_artifact(ctx, reg, run_id)
    labels = {"test_pr_auc_x1000": x1000(new),
              "test_roc_auc_x1000": x1000(card["test_metrics"]["roc_auc"]),
              "champion_pr_auc_x1000": x1000(old),
              "champion_version": label(champion_version or "none"),
              "promoted": label(promoted),
              "candidate": label(cand["name"]),
              "data_snapshot": label(card["data_version"]["snapshot"]),
              "run_id": label(run_id),
              "git_sha": label(os.environ.get("GIT_SHA", "manual")[:12])}
    model = aiplatform.Model.upload(
        display_name=reg["model_name"],
        artifact_uri=uri,
        serving_container_image_uri=serving_image(reg),
        serving_container_predict_route=reg["predict_route"],
        serving_container_health_route=reg["health_route"],
        serving_container_ports=[int(reg["serving_port"])],
        parent_model=parent.resource_name if parent else None,
        # the default version follows the champion
        is_default_version=promoted,
        labels=labels,
        description=f"{cand['name']}, test PR-AUC {new:.3f} (champion "
                    f"{'none' if old is None else f'{old:.3f}'} on the same "
                    f"rows), data {card['data_version']['snapshot']}")
    version = str(model.version_id)
    registry = registry_for(aiplatform, parent or model)
    move_alias(registry, CHALLENGER, version)
    if promoted:
        make_champion(registry, version)

    result = {"model": model.resource_name, "version": version,
              "artifact_uri": uri, "serving_image": serving_image(reg),
              "promoted": promoted, "reason": reason,
              "new_test_pr_auc": new,
              "champion_version_before": champion_version,
              "champion_test_pr_auc_same_rows": old,
              "compared_on": {"snapshot": card["data_version"]["snapshot"],
                              "n_test": int(len(test))},
              "min_improvement": gate["min_improvement"]}
    ctx.write_json("reports/registry.json", result)
    tracking.set_tags(ctx, {"registry_version": version,
                            "promoted": str(promoted)})
    tracking.log_metrics(ctx, {"champion_test_pr_auc_same_rows": old,
                               "challenger_test_pr_auc": new})

    old_text = "none" if old is None else f"{old:.3f}"
    if promoted:
        notify.alert(ctx, f"model version {version} promoted to champion",
                     f"{cand['name']}: test PR-AUC {new:.3f}; previous "
                     f"champion (version {champion_version}) on the same "
                     f"test rows: {old_text}. {reason}.")
    else:
        notify.alert(ctx, f"model version {version} NOT promoted",
                     f"{reason}. The champion stays at version "
                     f"{champion_version}. To promote anyway: python -m "
                     f"model_pipeline.registry rollback --version {version}")
    logger.info("registered version %s, promoted=%s (%s)", version,
                promoted, reason)
    return result


def status(reg):
    aiplatform = _aiplatform(reg)
    parent = find_parent(aiplatform, reg["model_name"])
    if parent is None:
        print("no model registered yet")
        return
    for v in registry_for(aiplatform, parent).list_versions():
        m = aiplatform.Model(f"{parent.resource_name}@{v.version_id}")
        lab = m.labels or {}
        print(f"version {v.version_id}: aliases {list(v.version_aliases)}, "
              f"test PR-AUC x1000 {lab.get('test_pr_auc_x1000', '?')}, "
              f"champion then {lab.get('champion_pr_auc_x1000', '?')}, "
              f"data {lab.get('data_snapshot', '?')}, "
              f"created {v.version_create_time}")


def rollback(reg, version):
    """Move the champion alias (and the default version) to `version`."""
    aiplatform = _aiplatform(reg)
    parent = find_parent(aiplatform, reg["model_name"])
    if parent is None:
        raise SystemExit("no model registered yet")
    registry = registry_for(aiplatform, parent)
    known = {str(v.version_id) for v in registry.list_versions()}
    if str(version) not in known:
        raise SystemExit(f"no version {version}; versions: {sorted(known)}")
    before = holder_of(registry, CHAMPION)
    make_champion(registry, version)
    print(f"champion is now version {version} (was {before})")
    return before


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["status", "rollback"])
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    params = load_params()
    reg = params["registry"]
    if args.command == "status":
        status(reg)
        return
    if not args.version:
        parser.error("rollback needs --version")
    before = rollback(reg, args.version)
    ctx = Context(tempfile.mkdtemp(prefix="ctt-rollback-"), params)
    notify.alert(ctx, f"manual rollback to version {args.version}",
                 f"The champion alias was moved by hand from version "
                 f"{before} to {args.version}.")


if __name__ == "__main__":
    main()
