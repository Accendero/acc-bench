"""Problems a first-time user hit in a cold read of the user guide, and their fixes."""

import numpy as np
import pytest
import yaml

from accbench import claims
from accbench.cli import main
from accbench.resolution import resolve
from accbench.rules import RuleError, SyntheticContext, check_questions, decide, k_of_n_cells
from accbench.runner import run_grid


def _questions(bench, questions):
    data = {"grid": "grid.yaml", "resolution": "resolution.json", "questions": questions}
    path = bench.root / "questions.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _q(params, rule="difference_clears_floor"):
    return {"id": "Q1", "question": "?", "rule": rule, "params": params}


# ---------------------------------------------------------------- questions are checked early


def test_wrong_cell_name_is_refused_before_any_run(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    q = _questions(bench, [_q({"cell": "Phase1", "a": "logreg", "b": "hist_gbm"})])
    with pytest.raises(RuleError, match="'Phase1' is not a cell; the cells are"):
        check_questions(q)


def test_missing_and_unknown_parameters_are_refused(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    q = _questions(
        bench,
        [
            _q(
                {"cell_list": ["mortality/Phase1"], "a": "logreg", "b": "hist_gbm", "k": 1},
                rule="k_of_n_cells",
            )
        ],
    )
    with pytest.raises(RuleError) as err:
        check_questions(q)
    assert "needs parameter 'cells'" in str(err.value)
    assert "has no parameter 'cell_list'" in str(err.value)


def test_unknown_method_is_refused(bench):
    bench.build(methods=["logreg"])
    q = _questions(
        bench,
        [_q({"cell": "mortality/Phase1", "a": "logreg", "b": {"select": ["logreg", "xgboost"]}})],
    )
    with pytest.raises(RuleError, match="method 'xgboost' is not in the grid"):
        check_questions(q)


def test_a_claim_cannot_be_registered_on_a_broken_question(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    _questions(bench, [_q({"cell": "south", "a": "logreg", "b": "hist_gbm"})])
    with pytest.raises(RuleError, match="not a cell"):
        claims.register(
            bench.root / "claims.jsonl",
            claim_id="C1",
            statement="s",
            question="Q1",
            prediction="effect",
            questions=bench.root / "questions.yaml",
            construct=bench.root / "construct.yaml",
        )


def test_rules_test_cli_checks_the_questions(bench, capsys):
    bench.build(methods=["logreg", "hist_gbm"])
    good = _questions(bench, [_q({"cell": "mortality/Phase1", "a": "logreg", "b": "hist_gbm"})])
    assert main(["rules", "test", str(good), "--rule", "difference_clears_floor"]) == 0
    assert "match the grid" in capsys.readouterr().out
    bad = _questions(bench, [_q({"cell": "south", "a": "logreg", "b": "hist_gbm"})])
    assert main(["rules", "test", str(bad)]) == 1
    assert "not a cell" in capsys.readouterr().err


# ---------------------------------------------------------------- first results cannot move later


def _flow(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    _questions(
        bench, [_q({"cell": "mortality/Phase1", "a": "logreg", "b": "hist_gbm", "n_boot": 100})]
    )
    run_grid(bench.grid)
    resolve(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")


def _touch(bench):
    f = bench.root / "my_methods.py"
    f.write_text(f.read_text() + "\n# changed\n")


def test_rerunning_everything_does_not_make_a_late_claim_early(bench):
    _flow(bench)
    claims.register(
        bench.root / "claims.jsonl",
        claim_id="C1",
        statement="s",
        question="Q1",
        prediction="effect",
        questions=bench.root / "questions.yaml",
        construct=bench.root / "construct.yaml",
    )
    _touch(bench)
    run_grid(bench.grid, on_stale="rerun")  # every record is now newer than the claim
    resolve(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")
    with pytest.raises(claims.ClaimError, match="after the first run"):
        claims.resolve(bench.root / "claims.jsonl", verdicts=bench.root / "verdicts.json")


def test_amendment_after_seen_results_is_flagged_after_a_rerun(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    q = [
        _q({"cell": "mortality/Phase1", "a": "logreg", "b": "hist_gbm", "n_boot": 100}),
        {
            **_q(
                {
                    "cells": ["mortality/Phase1"],
                    "a": "logreg",
                    "b": "hist_gbm",
                    "k": 1,
                    "n_boot": 100,
                },
                rule="k_of_n_cells",
            ),
            "id": "Q2",
        },
    ]
    _questions(bench, q)
    claims.register(
        bench.root / "claims.jsonl",
        claim_id="C1",
        statement="s",
        question="Q1",
        prediction="effect",
        questions=bench.root / "questions.yaml",
        construct=bench.root / "construct.yaml",
    )
    run_grid(bench.grid)
    resolve(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")  # the results are seen here
    c = yaml.safe_load((bench.root / "construct.yaml").read_text())
    c["version"] = 2
    c["amendments"] = [{"version": 2, "date": "2026-11-01", "reason": "Use Q2."}]
    (bench.root / "construct.yaml").write_text(yaml.safe_dump(c))
    claims.amend(
        bench.root / "claims.jsonl",
        claim_id="C1",
        reason="Use Q2.",
        construct=bench.root / "construct.yaml",
        question="Q2",
        questions=bench.root / "questions.yaml",
    )
    from accbench.channels import write_manifest
    from accbench.fixtures import freeze

    fx = freeze(bench.spec_path, out_dir=bench.out)
    write_manifest(bench.root / "channels.yaml", fx, bench.root / "channel_manifest.json")
    bench.grid_data["fixture"] = fx.path.relative_to(bench.root).as_posix()
    bench.write_grid()
    run_grid(bench.grid, on_stale="rerun")  # new records, all after the amendment
    resolve(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")
    [entry] = claims.resolve(bench.root / "claims.jsonl", verdicts=bench.root / "verdicts.json")
    assert entry["amended_after_results"] is True


def test_history_log_records_every_run(bench):
    bench.build(methods=["logreg"])
    run_grid(bench.grid)
    _touch(bench)
    run_grid(bench.grid, on_stale="rerun")
    from accbench.appendlog import read_log

    entries = read_log(bench.root / "runs.history.jsonl")
    assert len(entries) == 6 and len({e["code_state"] for e in entries}) == 2


# ---------------------------------------------------------------- k of n when cells cannot decide


def _cells_ctx(kinds, floor=0.03, seed=0):
    """kinds: 'win' (a far better), 'same' (identical scores) or 'noisy' (equal skill,
    independent noise on few rows, so the interval is wide)."""
    rng = np.random.default_rng(seed)
    rows, floors = {}, {}
    for i, kind in enumerate(kinds):
        n = 80 if kind == "noisy" else 300
        y = rng.integers(0, 2, n)
        c = np.array([f"t{j}" for j in range(n)])
        base = y * 1.0 + rng.normal(0, 1, n)
        if kind == "win":
            a, b = base + 3.0 * y, base
        elif kind == "same":
            a, b = base, base.copy()
        else:
            a, b = y + rng.normal(0, 1, n), y + rng.normal(0, 1, n)
        rows[(f"c{i}", "a")] = (y, a, c, None)
        rows[(f"c{i}", "b")] = (y, b, c, None)
        floors[f"c{i}"] = floor
    return SyntheticContext(rows, floors), [f"c{i}" for i in range(len(kinds))]


def test_k_of_n_is_inconclusive_when_undecided_cells_could_reach_k():
    ctx, cells = _cells_ctx(["win", "noisy"])
    v = k_of_n_cells(ctx, cells=cells, a="a", b="b", k=2, n_boot=200)
    assert v.outcome == "inconclusive" and v.details["clears"] == ["c0"]


def test_k_of_n_is_no_effect_when_k_is_out_of_reach():
    ctx, cells = _cells_ctx(["same", "same"])
    v = k_of_n_cells(ctx, cells=cells, a="a", b="b", k=1, n_boot=200)
    assert v.outcome == "no_effect" and v.details["unresolved"] == []


# ---------------------------------------------------------------- what the commands show


def test_freeze_shows_each_cell(project, capsys):
    project.write()
    assert main(["fixtures", "freeze", str(project.spec_path), "--out", str(project.out)]) == 0
    out = capsys.readouterr().out
    assert "mortality/Phase1" in out and "validation" in out


# ---------------------------------------------------------------- second cold read


def _resolved(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    _questions(
        bench,
        [
            _q(
                {
                    "cell": "mortality/Phase1",
                    "a": "logreg",
                    "b": {"select": ["logreg", "hist_gbm"]},
                    "n_boot": 100,
                }
            )
        ],
    )
    for cid in ("C1", "C2"):
        claims.register(
            bench.root / "claims.jsonl",
            claim_id=cid,
            statement="s",
            question="Q1",
            prediction="effect",
            questions=bench.root / "questions.yaml",
            construct=bench.root / "construct.yaml",
        )
    run_grid(bench.grid)
    resolve(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")
    claims.resolve(bench.root / "claims.jsonl", verdicts=bench.root / "verdicts.json")
    doc = bench.root / "report.md"
    doc.write_text("A finding [claim:C1].\n")
    return doc


def test_check_doc_refuses_results_from_old_code(bench):
    doc = _resolved(bench)
    assert claims.check_document(bench.root / "claims.jsonl", doc)["uncited"] == ["C2"]
    _touch(bench)  # a code change after the results
    with pytest.raises(claims.ClaimError, match="C1: its verdicts are out of date"):
        claims.check_document(bench.root / "claims.jsonl", doc)


def test_check_doc_require_all(bench):
    doc = _resolved(bench)
    with pytest.raises(claims.ClaimError, match="does not cite"):
        claims.check_document(bench.root / "claims.jsonl", doc, require_all=True)


def test_verdict_shows_the_selected_method(bench, capsys, monkeypatch):
    _resolved(bench)
    assert main(["verdict", str(bench.root / "questions.yaml")]) == 0
    assert "selected" in capsys.readouterr().out


def test_stale_grid_message_names_the_fix(bench):
    from accbench.runner import IncompleteGrid

    bench.build(methods=["logreg"])
    run_grid(bench.grid)
    _touch(bench)
    with pytest.raises(IncompleteGrid, match="run acc-bench run --rerun-stale"):
        resolve(bench.grid, n_boot=50)


# ---------------------------------------------------------------- third cold read


def _two_questions(bench):
    bench.build(methods=["logreg", "hist_gbm"])
    return _questions(
        bench,
        [
            _q(
                {
                    "cell": "mortality/Phase1",
                    "a": "logreg",
                    "b": {"select": ["logreg", "hist_gbm"]},
                    "n_boot": 100,
                }
            ),
            {
                **_q(
                    {
                        "cells": ["mortality/Phase1", "mortality/Phase2"],
                        "a": "logreg",
                        "b": "hist_gbm",
                        "k": 1,
                        "n_boot": 100,
                    },
                    rule="k_of_n_cells",
                ),
                "id": "Q2",
            },
        ],
    )


def _register(bench, cid, question, prediction):
    return claims.register(
        bench.root / "claims.jsonl",
        claim_id=cid,
        statement="s",
        question=question,
        prediction=prediction,
        questions=bench.root / "questions.yaml",
        construct=bench.root / "construct.yaml",
    )


def test_direction_on_a_rule_without_one_is_refused_when_registered(bench):
    _two_questions(bench)
    with pytest.raises(claims.ClaimError, match="does not report which method is better"):
        _register(bench, "C1", "Q2", "effect:a")
    with pytest.raises(claims.ClaimError, match="not a or b"):
        _register(bench, "C1", "Q1", "effect:xgboost")
    _register(bench, "C1", "Q1", "effect:b")


def test_side_predictions_follow_the_selected_method():
    v = {
        "outcome": "effect",
        "rule": "difference_clears_floor",
        "details": {"a": "tfidf", "b": "hist_gbm", "better": "hist_gbm"},
    }
    assert claims.status_for("effect:b", v) == "confirmed"
    assert claims.status_for("effect:a", v) == "inverted"


def test_amend_before_the_new_freeze_and_with_a_new_statement(bench):
    _two_questions(bench)
    _register(bench, "C1", "Q1", "effect:a")
    c = yaml.safe_load((bench.root / "construct.yaml").read_text())
    c["version"] = 2
    c["amendments"] = [{"version": 2, "date": "2026-11-01", "reason": "Use Q2."}]
    (bench.root / "construct.yaml").write_text(yaml.safe_dump(c))
    # The fixture is not frozen under version 2 yet; the amendment must still work.
    with pytest.raises(claims.ClaimError, match="does not report which method is better"):
        claims.amend(
            bench.root / "claims.jsonl",
            claim_id="C1",
            reason="Use Q2.",
            construct=bench.root / "construct.yaml",
            question="Q2",
            questions=bench.root / "questions.yaml",
        )
    claims.amend(
        bench.root / "claims.jsonl",
        claim_id="C1",
        reason="Use Q2.",
        construct=bench.root / "construct.yaml",
        question="Q2",
        prediction="effect",
        statement="logreg beats hist_gbm in a phase.",
        questions=bench.root / "questions.yaml",
    )
    row = claims.table(bench.root / "claims.jsonl")[0]
    assert row["test"] == "Q2" and row["statement"] == "logreg beats hist_gbm in a phase."
