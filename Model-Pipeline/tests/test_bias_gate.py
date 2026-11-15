"""Tests for the bias gate: gaps and intervals, the insufficient-data case,
thresholds fitted on validation scores, ranking and the test-set fallback."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from model_pipeline import bias, gaps, select, test_set
from model_pipeline.config import Context, load_params

B = {"metrics": ["recall"], "min_group_size": 5, "min_stopped": 5,
     "min_groups": 2, "bootstrap": 200, "ci_level": 0.9}
GATE = {"min_pr_auc": 0.3, "min_roc_auc": 0.6, "alert_bias_gap": 0.1,
        "block_bias_gap": 0.25, "bias_penalty": 0.5,
        "insufficient_bias_data": "block", "max_test_tries": 2}


def groups_data(sizes):
    """Labels, scores and groups: {group: (trials, stopped, caught)}; the
    caught stopped trials score 0.9, all others 0.1."""
    y, s, g = [], [], []
    for name, (n, stopped, caught) in sizes.items():
        y += [1] * stopped + [0] * (n - stopped)
        s += [0.9] * caught + [0.1] * (n - caught)
        g += [name] * n
    return np.array(y), np.array(s), np.array(g)


def check(sizes, b=B, gate=GATE):
    y, s, g = groups_data(sizes)
    rows = pd.DataFrame({"label": y, "region": g})
    return gaps.assess(rows, {"m": s}, ["region"], ["region"], 0.5, b,
                       gate, seed=0)


def test_gap_is_worst_group_against_overall():
    _, slices, d = check({"A": (400, 200, 180), "B": (400, 200, 120),
                          "C": (400, 200, 170)}, {**B, "bootstrap": 0})
    overall = (180 + 120 + 170) / 600
    assert d["m"]["max_gap"] == pytest.approx(overall - 0.6)
    assert slices.iloc[0]["worst_group"] == "B"
    assert d["m"]["slice"] == "region" and d["m"]["metric"] == "recall"


def test_no_qualified_groups_is_insufficient_data_not_a_pass():
    sizes = {"A": (8, 3, 3), "B": (8, 3, 0)}       # under min_stopped
    _, slices, d = check(sizes)
    assert slices.iloc[0]["status"] == "insufficient data"
    assert d["m"]["status"] == "insufficient data" and d["m"]["blocked"]
    _, _, d = check(sizes, gate={**GATE, "insufficient_bias_data": "alert"})
    assert d["m"]["status"] == "insufficient data"
    assert not d["m"]["blocked"]


def test_one_qualified_group_is_not_enough():
    _, slices, d = check({"A": (400, 200, 180), "B": (8, 3, 0)})
    assert slices.iloc[0]["n_qualified"] == 1
    assert d["m"]["status"] == "insufficient data"


def test_large_gap_in_small_group_does_not_block_but_in_large_group_does():
    # Same recall 0.4 in group B (gap above the block level either way).
    _, small, d_small = check({"A": (4000, 2000, 1800), "B": (5, 5, 2)})
    _, large, d_large = check({"A": (4000, 2000, 1800),
                               "B": (1000, 500, 200)})
    assert small.iloc[0]["gap"] > GATE["block_bias_gap"]
    assert small.iloc[0]["ci_low"] < GATE["block_bias_gap"]
    assert not d_small["m"]["blocked"]
    assert large.iloc[0]["ci_low"] > GATE["block_bias_gap"]
    assert d_large["m"]["blocked"] and d_large["m"]["status"] == "block"


def test_report_slice_alerts_but_never_blocks():
    y, s, g = groups_data({"A": (4000, 2000, 1800), "B": (1000, 500, 200)})
    rows = pd.DataFrame({"label": y, "region": "X", "area": g})
    _, slices, d = gaps.assess(rows, {"m": s}, ["region", "area"],
                               ["region"], 0.5, B, GATE, seed=0)
    area = slices[slices["slice"] == "area"].iloc[0]
    assert area["status"] == "alert" and area["role"] == "report"
    # region has a single group, so no gating slice could be checked
    assert d["m"]["status"] == "insufficient data"


def test_pr_auc_matches_sklearn_with_ties_and_weights():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.3).astype(float)
    s = np.round(rng.random(300) * (1 + y), 1)
    order = np.argsort(-s, kind="mergesort")
    counts = gaps.resample_counts(300, 1, 0)
    idx = np.repeat(np.arange(300), counts[0].astype(int))
    got = gaps.pr_auc(y[order], s[order],
                      np.vstack([np.ones(300), counts[:, order]]))
    assert got[0] == pytest.approx(average_precision_score(y, s))
    assert got[1] == pytest.approx(average_precision_score(y[idx], s[idx]))


def test_crossfit_scores_each_fold_with_cuts_from_the_other():
    rng = np.random.default_rng(0)
    n = 400
    groups = rng.choice(["a", "b"], n)
    y = (rng.random(n) < 0.4).astype(int)
    s = np.clip(rng.random(n) * 0.6 + 0.3 * y + 0.1 * (groups == "a"),
                0.01, 0.99)
    slices = {"region": groups}
    oof, steps = bias.crossfit_thresholds(s, y, slices, ["region"], 0.5, 5,
                                          2, seed=1)
    # the cuts kept for deployment are fitted on all trials
    assert steps[0]["cuts"] == bias.fit_threshold_steps(
        s, y, slices, ["region"], 0.5, 5)[0]["cuts"]
    # each fold is scored with cuts fitted on the other fold only
    folds = bias.fold_ids(y, groups, 2, 1)
    held = folds == 0
    fit_part = bias.fit_threshold_steps(s[~held], y[~held],
                                        {"region": groups[~held]},
                                        ["region"], 0.5, 5)
    expect = bias.apply_threshold_steps(s[held], {"region": groups[held]},
                                        fit_part, 0.5)
    assert np.allclose(oof[held], expect)
    assert not np.allclose(oof, bias.apply_threshold_steps(s, slices, steps,
                                                           0.5))


class FixedModel:
    """Model whose score is the first feature column."""

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def make_ctx(tmp_path, params=None):
    params = params or load_params()
    params["bias"].update(gating_slices=["region"], report_slices=[],
                          metrics=["recall"], min_group_size=20,
                          min_stopped=5, bootstrap=50,
                          threshold_slices=[["region"]])
    ctx = Context(tmp_path / "work", params)
    ctx.write_json("data/columns.json", ["f0"])
    ctx.write_json("data/data_version.json", {"slices": ["region"]})
    joblib.dump(FixedModel(), ctx.path("models", "m.joblib"))
    ctx.write_json("candidates.json", [{
        "name": "m", "family": "m", "mitigation": "none", "params": {},
        "model_file": "models/m.joblib"}])
    return ctx


def split(n, seed, invert=False):
    rng = np.random.default_rng(seed)
    region = rng.choice(["US", "EU"], n)
    y = (rng.random(n) < 0.4).astype(int)
    lift = np.where(region == "US", 0.25, 0.0)
    if invert:       # train scores sit far higher for EU than validation
        lift = np.where(region == "US", 0.0, 0.4)
    f0 = np.clip(0.2 + 0.35 * y + lift + rng.normal(0, 0.1, n), 0.01, 0.99)
    return pd.DataFrame({"nct_id": [f"N{seed}{i}" for i in range(n)],
                         "label": y, "f0": f0, "region": region})


def test_thresholds_are_fitted_on_validation_scores(tmp_path, monkeypatch):
    monkeypatch.setattr("model_pipeline.notify.alert", lambda *a, **k: None)
    for name in ("log_artifacts", "log_metrics"):
        monkeypatch.setattr(f"model_pipeline.tracking.{name}",
                            lambda *a, **k: None)
    ctx = make_ctx(tmp_path)
    train, val = split(600, 1, invert=True), split(600, 2)
    train.to_pickle(ctx.path("data", "train.pkl"))
    val.to_pickle(ctx.path("data", "val.pkl"))
    ctx.write_json("reports/val_metrics.json", {})
    bias.run(ctx)

    cand = next(c for c in ctx.read_json("candidates.json")
                if c["mitigation"] == "group_thresholds")
    assert cand["thresholds_fitted_on"] == "validation"
    slices_v = {"region": val["region"].to_numpy()}
    on_val = bias.fit_threshold_steps(val["f0"].to_numpy(),
                                      val["label"].to_numpy(), slices_v,
                                      ["region"], 0.5, 5)
    on_train = bias.fit_threshold_steps(
        train["f0"].to_numpy(), train["label"].to_numpy(),
        {"region": train["region"].to_numpy()}, ["region"], 0.5, 5)
    assert cand["threshold_steps"][0]["cuts"] == pytest.approx(
        on_val[0]["cuts"])
    assert cand["threshold_steps"][0]["cuts"] != pytest.approx(
        on_train[0]["cuts"])
    # validation metrics of the threshold candidate use out-of-fold scores
    scores = pd.read_csv(ctx.path("reports", "bias", "val_scores.csv"))
    in_sample = bias.apply_threshold_steps(val["f0"].to_numpy(), slices_v,
                                           on_val, 0.5)
    assert not np.allclose(scores[cand["name"]], in_sample)
    assert (tmp_path / "work/reports/bias/bias_report.md").exists()


def test_choose_ranks_eligible_by_penalised_score():
    rows = {"a": {"pr_auc": 0.50, "roc_auc": 0.8, "max_gap": 0.30,
                  "bias_blocked": True, "bias_status": "block"},
            "b": {"pr_auc": 0.46, "roc_auc": 0.8, "max_gap": 0.05},
            "c": {"pr_auc": 0.48, "roc_auc": 0.8, "max_gap": 0.20},
            "d": {"pr_auc": 0.25, "roc_auc": 0.8, "max_gap": 0.0},
            "e": {"pr_auc": 0.40, "roc_auc": 0.8, "max_gap": 0.0,
                  "bias_blocked": True, "bias_status": "insufficient data"}}
    winner, table = select.choose(rows, GATE)
    assert [t["name"] for t in table if t["eligible"]] == ["b", "c"]
    assert winner["name"] == "b" and winner["rank"] == 1
    assert table[1]["name"] == "c" and table[1]["rank"] == 2
    reasons = {t["name"]: t["reasons"] for t in table}
    assert reasons["e"] == ["bias: insufficient data"]
    assert any("PR-AUC" in r for r in reasons["d"])


def fallback_ctx(tmp_path, monkeypatch, failing):
    """Context with three eligible candidates; evaluate() fails the ones in
    `failing` on the test set."""
    for name in ("log_artifacts", "log_metrics", "set_tags"):
        monkeypatch.setattr(f"model_pipeline.tracking.{name}",
                            lambda *a, **k: None)
    sent = []
    monkeypatch.setattr("model_pipeline.notify.alert",
                        lambda ctx, subject, body: sent.append(subject))
    ctx = make_ctx(tmp_path)
    test = split(100, 3)
    test.to_pickle(ctx.path("data", "test.pkl"))
    names = ["x", "y", "z"]
    ctx.write_json("candidates.json", [
        {"name": n, "family": "m", "mitigation": "none",
         "model_file": "models/m.joblib"} for n in names])
    ctx.write_json("reports/val_metrics.json", {n: {"pr_auc": 0.5}
                                                for n in names})
    table = [{"name": n, "pr_auc": 0.5, "roc_auc": 0.8, "max_gap": 0.0,
              "score": 0.5 - i / 10, "eligible": True, "rank": i + 1,
              "reasons": [], "bias_status": "ok"}
             for i, n in enumerate(names)]
    ctx.write_json("reports/selection.json",
                   {"selected": table[0], "candidates": table})

    # the calibrate stage calibrated the top candidate; a fallback one is
    # calibrated by the test stage (checked in its own test below)
    ctx.write_json("reports/calibration/calibration.json",
                   {"candidate": "x", "method": "isotonic"})
    joblib.dump(None, ctx.path("models", "calibrator.joblib"))
    monkeypatch.setattr(test_set, "calibrator_for", lambda ctx, cand: None)
    monkeypatch.setattr("model_pipeline.calibrate.apply",
                        lambda calibrator, s: s)

    def fake_evaluate(ctx, cand, df):
        bad = cand["name"] in failing
        return {"scores": df["f0"].to_numpy(), "columns": ["f0"],
                "metrics": {"pr_auc": 0.2 if bad else 0.5, "roc_auc": 0.8,
                            "max_bias_gap": 0.0, "bias_status": "ok"},
                "table": pd.DataFrame(), "groups": pd.DataFrame(),
                "slices": pd.DataFrame(columns=["metric", "model"]),
                "decision": {"status": "ok", "blocked": False,
                             "alerts": [], "blocks": []},
                "problems": ["test PR-AUC 0.200 < 0.3"] if bad else []}

    monkeypatch.setattr(test_set, "evaluate", fake_evaluate)
    return ctx, sent


def test_test_stage_falls_back_to_next_candidate(tmp_path, monkeypatch):
    ctx, sent = fallback_ctx(tmp_path, monkeypatch, failing={"x"})
    test_set.run(ctx)
    card = ctx.read_json("artifact/model_card.json")
    assert card["candidate"]["name"] == "y"
    assert [t["name"] for t in card["test_tries"]] == ["x", "y"]
    selection = ctx.read_json("reports/selection.json")
    assert selection["selected"]["name"] == "y"
    assert selection["selected_on_validation"]["name"] == "x"
    assert any("fell back to y" in s for s in sent)
    assert "Final model: y" in ctx.path("reports", "selection.md") \
        .read_text()


def test_test_stage_stops_when_no_tried_candidate_passes(tmp_path,
                                                         monkeypatch):
    # z would pass but max_test_tries is 2, so only x and y are tried
    ctx, sent = fallback_ctx(tmp_path, monkeypatch, failing={"x", "y"})
    ctx.params["gate"]["max_test_tries"] = 2
    with pytest.raises(SystemExit):
        test_set.run(ctx)
    tries = json.loads(ctx.path("reports", "test", "test_tries.json")
                       .read_text())
    assert [t["name"] for t in tries] == ["x", "y"]
    assert not ctx.path("artifact", "model_card.json").exists()
    assert any("no candidate passed" in s for s in sent)


def test_a_fallback_candidate_gets_its_own_calibrator(tmp_path,
                                                      monkeypatch):
    ctx = make_ctx(tmp_path)
    ctx.write_json("reports/calibration/calibration.json",
                   {"candidate": "x", "method": "isotonic"})
    joblib.dump("calibrator of x", ctx.path("models", "calibrator.joblib"))
    refit = []

    def fake_calibrate(ctx, cand):
        refit.append(cand["name"])
        joblib.dump(f"calibrator of {cand['name']}",
                    ctx.path("models", "calibrator.joblib"))

    monkeypatch.setattr("model_pipeline.calibrate.calibrate", fake_calibrate)
    assert test_set.calibrator_for(ctx, {"name": "x"}) == "calibrator of x"
    assert refit == []
    assert test_set.calibrator_for(ctx, {"name": "y"}) == "calibrator of y"
    assert refit == ["y"]
