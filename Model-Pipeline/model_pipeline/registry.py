"""Stage register: push the model to the Vertex AI Model Registry, promote
it only if it beats the current champion, and roll back by hand if needed.

Every run that reaches this stage uploads a new version of
ctt-termination-model with the alias "challenger" and its metrics and data
version as labels. The alias "champion" (the model the inference service
loads) moves to the new version only if its test PR-AUC is at least the
champion's plus min_improvement. Otherwise the champion stays: a worse
model is never promoted, and an alert says so.

Manual rollback or a look at the versions:
    python -m model_pipeline.registry status
    python -m model_pipeline.registry rollback --version 3
"""

import argparse
import logging
import os

from model_pipeline import notify, tracking
from model_pipeline.config import Context, load_params

logger = logging.getLogger(__name__)
CHAMPION = "champion"


def label(value):
    """Vertex AI label values: lowercase letters, digits, - and _."""
    text = str(value).lower()
    clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in text)
    return clean[:63] or "none"


def should_promote(new_score, champion_score, min_improvement):
    """True if there is no champion yet or the new model is not worse."""
    if champion_score is None:
        return True
    return new_score >= champion_score + min_improvement


def champion_score(model):
    """Test PR-AUC stored on a registry version (labels keep x1000)."""
    value = (model.labels or {}).get("test_pr_auc_x1000")
    return int(value) / 1000 if value else None


def _aiplatform(reg):
    from google.cloud import aiplatform
    aiplatform.init(project=reg["project"], location=reg["region"])
    return aiplatform


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


def upload_artifact(ctx, reg, run_id):
    import gcsfs
    bucket = ctx.params["data"]["bucket"]
    dest = f"{bucket}/{reg['artifacts_prefix']}/{run_id}"
    fs = gcsfs.GCSFileSystem()
    for f in ctx.path("artifact").iterdir():
        if f.is_file() and f.name != ".keep":
            fs.put_file(str(f), f"{dest}/{f.name}")
    return f"gs://{dest}"


def run(ctx):
    from google.cloud.aiplatform.models import ModelRegistry

    reg, gate = ctx.params["registry"], ctx.params["gate"]
    card = ctx.read_json("artifact/model_card.json")
    cand, test = card["candidate"], card["test_metrics"]
    run_id = ctx.state.get("run_id", "manual")
    aiplatform = _aiplatform(reg)

    uri = upload_artifact(ctx, reg, run_id)
    parent = find_parent(aiplatform, reg["model_name"])
    champion = get_champion(aiplatform, parent)
    old = champion_score(champion) if champion else None

    labels = {"test_pr_auc_x1000": label(int(round(test["pr_auc"] * 1000))),
              "test_roc_auc_x1000": label(int(round(test["roc_auc"] * 1000))),
              "candidate": label(cand["name"]),
              "data_snapshot": label(card["data_version"]["snapshot"]),
              "run_id": label(run_id),
              "git_sha": label(os.environ.get("GIT_SHA", "manual")[:12])}
    new = aiplatform.Model.upload(
        display_name=reg["model_name"],
        artifact_uri=uri,
        serving_container_image_uri=reg["serving_images"][cand["family"]],
        parent_model=parent.resource_name if parent else None,
        is_default_version=parent is None,
        version_aliases=["challenger"],
        labels=labels,
        description=f"{cand['name']}, test PR-AUC {test['pr_auc']:.3f}, "
                    f"data {card['data_version']['snapshot']}")
    version = new.version_id
    promoted = should_promote(test["pr_auc"], old, gate["min_improvement"])
    if promoted:
        ModelRegistry(new.resource_name).add_version_aliases(
            [CHAMPION], version=version)
    result = {"model": new.resource_name, "version": version,
              "artifact_uri": uri, "promoted": promoted,
              "new_test_pr_auc": test["pr_auc"],
              "champion_test_pr_auc_before": old}
    ctx.write_json("reports/registry.json", result)
    tracking.set_tags(ctx, {"registry_version": version,
                            "promoted": str(promoted)})

    if promoted:
        notify.alert(ctx, f"model version {version} promoted to champion",
                     f"{cand['name']}: test PR-AUC {test['pr_auc']:.3f} "
                     f"(previous champion: {old}).")
    else:
        notify.alert(ctx, f"model version {version} NOT promoted",
                     f"Test PR-AUC {test['pr_auc']:.3f} is below the "
                     f"champion's {old:.3f}; the champion stays (rollback "
                     "to the previous model).")
    logger.info("registered version %s, promoted=%s", version, promoted)
    return result


def status(reg):
    aiplatform = _aiplatform(reg)
    parent = find_parent(aiplatform, reg["model_name"])
    if parent is None:
        print("no model registered yet")
        return
    versions = registry_for(aiplatform, parent).list_versions()
    for v in versions:
        print(f"version {v.version_id}: aliases {v.version_aliases}, "
              f"created {v.version_create_time}")


def registry_for(aiplatform, parent):
    from google.cloud.aiplatform.models import ModelRegistry
    return ModelRegistry(parent.resource_name)


def rollback(reg, version):
    """Move the champion alias back to an earlier version."""
    aiplatform = _aiplatform(reg)
    parent = find_parent(aiplatform, reg["model_name"])
    registry_for(aiplatform, parent).add_version_aliases(
        [CHAMPION], version=str(version))
    print(f"champion is now version {version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["status", "rollback"])
    parser.add_argument("--version")
    args = parser.parse_args()
    reg = load_params()["registry"]
    if args.command == "status":
        status(reg)
    else:
        if not args.version:
            parser.error("rollback needs --version")
        rollback(reg, args.version)
        ctx = Context("/tmp/ctt-rollback", load_params())
        notify.alert(ctx, f"manual rollback to version {args.version}",
                     "The champion alias was moved by hand.")


if __name__ == "__main__":
    main()
