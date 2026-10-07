import numpy as np
import pytest
import yaml

from accbench.cli import main
from accbench.errors import StaleInput
from accbench.provenance import read_artifact
from accbench.resampling import bootstrap_metric
from accbench.resolution import resolve
from accbench.rules import (
    NullInputFailure,
    RuleError,
    SyntheticContext,
    Verdict,
    _null_drops,
    decide,
    difference_clears_floor,
    ensure_null_suite,
    get_rule,
    rule,
    run_null_suite,
)
from accbench.runner import run_grid

BUILT_IN = ("difference_clears_floor", "k_of_n_cells", "difference_of_drops")


@pytest.fixture(scope="module")
def suite():
    return run_null_suite(BUILT_IN)


# ---------------------------------------------------------------- the null-input suite


def test_built_in_rules_pass_their_null_inputs(suite):
    # Other test modules register deliberately broken rules; only the built-ins are checked.
    assert set(suite) == set(BUILT_IN)
    for name, r in suite.items():
        assert r["passed"], (name, r)
    assert suite["difference_clears_floor"]["exact_null_claimed_effect"] is False


def _separate_drops_null(seed):
    return _null_drops(seed)


@rule("test_separate_drops", null_input=_separate_drops_null)
def separate_drops(ctx, *, cell, model, reference, seen_arm, unseen_arm, n_boot=200, seed=0):
    """The T28b defect: test each model's drop on its own, then compare the verdicts."""

    def drop_significant(method):
        y_s, s_s, c_s = ctx.test_rows(cell, method, seen_arm)
        y_u, s_u, c_u = ctx.test_rows(cell, method, unseen_arm)
        seen = bootstrap_metric(ctx.goodness, y_s, s_s, c_s, n_boot=n_boot, seed=seed)
        unseen = bootstrap_metric(ctx.goodness, y_u, s_u, c_u, n_boot=n_boot, seed=seed)
        return unseen.hi < seen.lo  # the intervals do not overlap

    if drop_significant(model) and not drop_significant(reference):
        return Verdict("effect", "outcome recall demonstrated")
    return Verdict("inconclusive", "no recall shown")


def test_null_suite_catches_the_separate_drops_defect(tmp_path):
    # Section 11 / T28b: the code asked whether each drop was significant on its own,
    # and declared memorization where the right answer was inconclusive.
    result = run_null_suite(["test_separate_drops"])["test_separate_drops"]
    assert not result["passed"] and result["effects_on_null"] > 3
    from accbench.provenance import code_state

    with pytest.raises(NullInputFailure, match="test_separate_drops"):
        ensure_null_suite(
            ["test_separate_drops"], tmp_path / "suite.json", code=code_state([tmp_path])
        )


def test_gate_refuses_a_small_suite(tmp_path):
    from accbench.provenance import code_state

    with pytest.raises(RuleError, match="at least 20"):
        ensure_null_suite(
            ["k_of_n_cells"], tmp_path / "s.json", code=code_state([tmp_path]), draws=5
        )


def _always_null(seed):
    return SyntheticContext({}, {}), {}


@rule("test_always_effect", null_input=_always_null, exact_null=_always_null)
def always_effect(ctx):
    return Verdict("effect", "always")


def test_exact_null_must_not_claim_an_effect():
    r = run_null_suite(["test_always_effect"], draws=2)["test_always_effect"]
    assert r["exact_null_claimed_effect"] is True and not r["passed"]


def test_suite_record_is_reused_until_the_rule_changes(tmp_path):
    from accbench.provenance import code_state

    code = code_state([tmp_path])
    path = tmp_path / "suite.json"
    ensure_null_suite(["difference_clears_floor"], path, code=code)
    mtime = path.stat().st_mtime_ns
    ensure_null_suite(["difference_clears_floor"], path, code=code)
    assert path.stat().st_mtime_ns == mtime
    # A different code state (a rule edited in the project) forces a rerun.
    (tmp_path / "my_rules.py").write_text("x = 1\n")
    ensure_null_suite(["difference_clears_floor"], path, code=code_state([tmp_path]))
    assert path.stat().st_mtime_ns != mtime


# ---------------------------------------------------------------- built-in rules on known data


def _pair_ctx(gap, n=400, seed=0, metric="roc_auc"):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = y * 2.0 + rng.normal(0, 1, n)
    b = y * (2.0 - gap) + rng.normal(0, 1, n)
    c = np.array([f"t{i}" for i in range(n)])
    rows = {("c", "a"): (y, a, c, None), ("c", "b"): (y, b, c, None)}
    return SyntheticContext(rows, {"c": 0.03}, {("c", "a"): 0.6, ("c", "b"): 0.8}, metric=metric)


def test_clear_difference_is_an_effect():
    v = difference_clears_floor(_pair_ctx(1.5), cell="c", a="a", b="b", n_boot=300)
    assert v.outcome == "effect" and v.details["better"] == "a"


def test_identical_methods_are_no_effect():
    ctx = _pair_ctx(0)
    ctx.rows[("c", "b")] = ctx.rows[("c", "a")]
    v = difference_clears_floor(ctx, cell="c", a="a", b="b", n_boot=200)
    assert v.outcome == "no_effect"


def test_lower_is_better_metrics_are_signed():
    v = difference_clears_floor(_pair_ctx(1.5, metric="brier"), cell="c", a="a", b="b", n_boot=200)
    assert v.details["difference"]["point"] != 0  # brier on raw scores, sign handled


def test_selection_uses_validation_only():
    ctx = _pair_ctx(1.5)
    # On validation, b looks better (0.8 > 0.6) although a is better on test.
    chosen = ctx.select("c", ["a", "b"])
    assert chosen == "b"
    assert ctx.tracking.selections[-1]["on"] == "validation" and not ctx.tracking.oracle


def test_oracle_selection_is_marked():
    ctx = _pair_ctx(1.5)
    assert ctx.oracle_select("c", ["a", "b"]) == "a"
    assert ctx.tracking.oracle and ctx.tracking.selections[-1]["on"] == "test (oracle)"


def test_unknown_outcome_and_rule():
    with pytest.raises(RuleError):
        Verdict("maybe", "")
    with pytest.raises(RuleError, match="unknown rule"):
        get_rule("nope")


# ---------------------------------------------------------------- questions to verdicts, end to end


def _bench_with_questions(bench, questions, *, rules_module=None):
    from conftest import make_trials

    bench.table = make_trials(1200, seed=4)
    bench.spec["cutoffs"] = {"model": "2022-01-01"}
    bench.spec["arms"] = {
        "seen": {"cutoff": "model", "before": ["results_posted_on"]},
        "unseen": {"cutoff": "model", "after": ["registered_on"]},
    }
    bench.build(methods=["logreg", "hist_gbm", "tfidf_logreg"], split_keys=["split-0"])
    run_grid(bench.grid)
    resolve(bench.grid, n_boot=50)
    data = {"grid": "grid.yaml", "resolution": "resolution.json", "questions": questions}
    if rules_module:
        data["rules_module"] = rules_module
    path = bench.root / "questions.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


QUESTIONS = [
    {
        "id": "Q1",
        "question": "Does TF-IDF beat the best tabular method on Phase 1?",
        "rule": "difference_clears_floor",
        "params": {
            "cell": "mortality/Phase1",
            "a": "tfidf_logreg",
            "b": {"select": ["logreg", "hist_gbm"]},
            "n_boot": 100,
        },
    },
    {
        "id": "Q2",
        "question": "Does logreg beat hist_gbm in 2 or more of 3 cells?",
        "rule": "k_of_n_cells",
        "params": {
            "cells": ["mortality/Phase1", "mortality/Phase2", "mortality/Phase3"],
            "a": "logreg",
            "b": "hist_gbm",
            "k": 2,
            "n_boot": 100,
        },
    },
    {
        "id": "Q3",
        "question": "Is logreg's advantage over hist_gbm larger on the seen arm?",
        "rule": "difference_of_drops",
        "params": {
            "cell": "mortality/Phase2",
            "model": "logreg",
            "reference": "hist_gbm",
            "seen_arm": "seen",
            "unseen_arm": "unseen",
            "n_boot": 100,
        },
    },
]


def test_decide_answers_every_question(bench):
    q = _bench_with_questions(bench, QUESTIONS)
    result = decide(q)
    assert [v["id"] for v in result["verdicts"]] == ["Q1", "Q2", "Q3"]
    for v in result["verdicts"]:
        assert v["outcome"] in ("effect", "no_effect", "inconclusive")
        assert len(v["rule_sha256"]) == 64 and v["oracle"] is False
    q1 = result["verdicts"][0]
    assert q1["selections"][0]["on"] == "validation"
    assert q1["selections"][0]["chosen"] in ("logreg", "hist_gbm")
    assert set(result["null_suite"]) == {
        "difference_clears_floor",
        "k_of_n_cells",
        "difference_of_drops",
    }
    art = read_artifact(bench.root / "verdicts.json")
    assert set(art.inputs) == {"questions", "resolution", "runs", "null_suite"}


def test_oracle_question_is_labelled(bench):
    questions = [dict(QUESTIONS[0])]
    questions[0]["params"] = {
        **QUESTIONS[0]["params"],
        "b": {"oracle_select": ["logreg", "hist_gbm"]},
    }
    result = decide(_bench_with_questions(bench, questions))
    assert result["verdicts"][0]["oracle"] is True


def test_decide_refuses_a_rule_that_fails_its_null(bench):
    module = (
        "from accbench.rules import rule, Verdict, SyntheticContext\n"
        "def null(seed):\n"
        "    return SyntheticContext({}, {}), {}\n"
        "@rule('always_yes', null_input=null)\n"
        "def always_yes(ctx, **params):\n"
        "    return Verdict('effect', 'yes')\n"
    )
    bench.root.mkdir(parents=True, exist_ok=True)
    (bench.root / "my_rules.py").write_text(module)
    questions = [{"id": "Q9", "question": "?", "rule": "always_yes", "params": {}}]
    q = _bench_with_questions(bench, questions, rules_module="my_rules.py")
    with pytest.raises(NullInputFailure, match="always_yes"):
        decide(q)
    assert not (bench.root / "verdicts.json").exists()


def test_decide_refuses_a_stale_resolution(bench):
    q = _bench_with_questions(bench, QUESTIONS[:1])
    (bench.root / "my_methods.py").write_text(
        (bench.root / "my_methods.py").read_text() + "\n# repaired\n"
    )
    run_grid(bench.grid, on_stale="rerun")
    with pytest.raises(StaleInput):
        decide(q)


def test_questions_validation(bench):
    bad = [{"id": "Q1", "question": "?", "rule": "nope", "params": {}}, {"id": "Q1"}]
    with pytest.raises(RuleError, match="unknown rule.*duplicate id"):
        decide(_bench_with_questions(bench, bad))


def test_cli_rules_test_and_verdict(bench, capsys):
    assert main(["rules", "test", "--rule", "difference_clears_floor"]) == 0
    assert "pass  difference_clears_floor" in capsys.readouterr().out
    q = _bench_with_questions(bench, QUESTIONS[:1])
    decide(q)  # writes the suite record so the CLI reuses it
    assert main(["verdict", str(q)]) == 0
    assert "Q1:" in capsys.readouterr().out


def test_few_draws_allow_no_effects():
    # A small suite must not let an always-effect rule through on its allowance.
    r = run_null_suite(["test_always_effect"], draws=3)["test_always_effect"]
    assert r["max_effects"] == 0 and not r["passed"]
    assert (
        run_null_suite(["difference_clears_floor"], draws=20)["difference_clears_floor"][
            "max_effects"
        ]
        == 3
    )
