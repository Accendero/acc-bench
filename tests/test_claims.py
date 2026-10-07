import json

import pytest
import yaml

from accbench import claims
from accbench.cli import main
from accbench.errors import LogError
from accbench.resolution import resolve as resolve_floors
from accbench.rules import decide
from accbench.runner import run_grid

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
]


def _setup(bench, questions=QUESTIONS):
    """Freeze, register the channels and write the questions; nothing has run yet."""
    bench.build(methods=["logreg", "hist_gbm", "tfidf_logreg"])
    data = {"grid": "grid.yaml", "resolution": "resolution.json", "questions": questions}
    (bench.root / "questions.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return bench


def _paths(bench):
    return {
        "log": bench.root / "claims.jsonl",
        "questions": bench.root / "questions.yaml",
        "construct": bench.root / "construct.yaml",
        "verdicts": bench.root / "verdicts.json",
    }


def _register(bench, claim_id, question, prediction, **kw):
    p = _paths(bench)
    return claims.register(
        p["log"],
        claim_id=claim_id,
        statement=f"claim {claim_id}",
        question=question,
        prediction=prediction,
        questions=p["questions"],
        construct=p["construct"],
        **kw,
    )


def _run_and_decide(bench):
    run_grid(bench.grid)
    resolve_floors(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")


# ---------------------------------------------------------------- the normal path


def test_register_run_resolve(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    _register(bench, "C2", "Q2", "no_effect")
    _register(bench, "C3", "Q2", "effect")
    p = _paths(bench)
    claims.mark_untested(p["log"], claim_id="C3", reason="Superseded by C2 before the run.")
    _run_and_decide(bench)

    resolved = claims.resolve(p["log"], verdicts=p["verdicts"])
    assert {e["claim_id"] for e in resolved} == {"C1", "C2"}
    rows = {r["claim"]: r for r in claims.table(p["log"])}
    for cid in ("C1", "C2"):
        assert rows[cid]["status"] in ("confirmed", "refuted", "inverted", "inconclusive")
        assert rows[cid]["test"] in ("Q1", "Q2") and rows[cid]["evidence"]
    assert rows["C3"]["status"] == "untested"
    assert rows["C3"]["untested_reason"].startswith("Superseded")
    assert sum(claims.counts(rows.values()).values()) == 3

    # Resolving again with the same verdicts adds nothing.
    assert claims.resolve(p["log"], verdicts=p["verdicts"]) == []


def test_status_follows_prediction_and_outcome():
    v = {"outcome": "effect", "rule": "r", "details": {"better": "a"}}
    assert claims.status_for("effect", v) == "confirmed"
    assert claims.status_for("effect:a", v) == "confirmed"
    assert claims.status_for("effect:b", v) == "inverted"  # T21: the prediction came back inverted
    assert claims.status_for("no_effect", v) == "refuted"
    assert claims.status_for("effect", {"outcome": "no_effect"}) == "refuted"
    assert claims.status_for("no_effect", {"outcome": "no_effect"}) == "confirmed"
    assert claims.status_for("effect", {"outcome": "inconclusive"}) == "inconclusive"
    with pytest.raises(claims.ClaimError, match="reports none"):
        claims.status_for("effect:a", {"outcome": "effect", "rule": "k", "details": {}})


# ---------------------------------------------------------------- registered before results


def test_claim_written_after_its_result_is_refused(bench):
    _setup(bench)
    _run_and_decide(bench)
    _register(bench, "C1", "Q1", "effect")  # written after the runs
    with pytest.raises(claims.ClaimError, match="after the first run"):
        claims.resolve(_paths(bench)["log"], verdicts=_paths(bench)["verdicts"])


def test_question_changed_after_registration_is_refused(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    changed = [dict(QUESTIONS[0]), QUESTIONS[1]]
    changed[0]["params"] = {**QUESTIONS[0]["params"], "a": "logreg", "b": "hist_gbm"}
    data = yaml.safe_load((bench.root / "questions.yaml").read_text())
    data["questions"] = changed
    (bench.root / "questions.yaml").write_text(yaml.safe_dump(data))
    _run_and_decide(bench)
    with pytest.raises(claims.ClaimError, match="has changed since the claim was registered"):
        claims.resolve(_paths(bench)["log"], verdicts=_paths(bench)["verdicts"])


def test_oracle_verdict_cannot_decide_a_confirmatory_claim(bench):
    oracle = [dict(QUESTIONS[0])]
    oracle[0]["params"] = {**QUESTIONS[0]["params"], "b": {"oracle_select": ["logreg", "hist_gbm"]}}
    _setup(bench, oracle)
    _register(bench, "C1", "Q1", "effect")
    _register(bench, "X1", "Q1", "effect", exploratory=True)
    _run_and_decide(bench)
    p = _paths(bench)
    with pytest.raises(claims.ClaimError, match="oracle"):
        claims.resolve(p["log"], verdicts=p["verdicts"])


def test_exploratory_claim_may_use_an_oracle(bench):
    oracle = [dict(QUESTIONS[0])]
    oracle[0]["params"] = {**QUESTIONS[0]["params"], "b": {"oracle_select": ["logreg", "hist_gbm"]}}
    _setup(bench, oracle)
    _register(bench, "X1", "Q1", "effect", exploratory=True)
    _run_and_decide(bench)
    p = _paths(bench)
    assert [e["claim_id"] for e in claims.resolve(p["log"], verdicts=p["verdicts"])] == ["X1"]


# ---------------------------------------------------------------- amendments in the open


def _bump_construct(bench, reason="Changed claim C1 to Q2."):
    c = yaml.safe_load((bench.root / "construct.yaml").read_text())
    c["version"] = c["version"] + 1
    c.setdefault("amendments", []).append(
        {"version": c["version"], "date": "2026-11-01", "reason": reason}
    )
    (bench.root / "construct.yaml").write_text(yaml.safe_dump(c, sort_keys=False))


def test_amendment_needs_the_construct_amended_first(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    p = _paths(bench)
    with pytest.raises(claims.ClaimError, match="construct statement amended first"):
        claims.amend(
            p["log"],
            claim_id="C1",
            reason="x",
            construct=p["construct"],
            question="Q2",
            questions=p["questions"],
        )
    _bump_construct(bench)
    e = claims.amend(
        p["log"],
        claim_id="C1",
        reason="Q2 is the pre-registered rule.",
        construct=p["construct"],
        question="Q2",
        questions=p["questions"],
    )
    assert e["construct_version"] == 2 and e["had_result"] is False
    row = claims.table(p["log"])[0]
    assert row["test"] == "Q2" and row["amendments"][0]["reason"].startswith("Q2")


def test_amendment_after_results_is_flagged(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    _run_and_decide(bench)
    p = _paths(bench)
    claims.resolve(p["log"], verdicts=p["verdicts"])
    _bump_construct(bench)
    claims.amend(
        p["log"],
        claim_id="C1",
        reason="Switched to the k-of-n question.",
        construct=p["construct"],
        question="Q2",
        questions=p["questions"],
    )
    assert claims.load_claims(p["log"])["C1"].status == "registered"
    [entry] = claims.resolve(p["log"], verdicts=p["verdicts"])
    assert entry["amended_after_results"] is True


def test_amendment_needs_a_reason_and_a_change(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    p = _paths(bench)
    _bump_construct(bench)
    with pytest.raises(claims.ClaimError, match="reason"):
        claims.amend(
            p["log"],
            claim_id="C1",
            reason=" ",
            construct=p["construct"],
            prediction="no_effect",
            questions=p["questions"],
        )
    with pytest.raises(claims.ClaimError, match="changes the prediction"):
        claims.amend(p["log"], claim_id="C1", reason="r", construct=p["construct"])


# ---------------------------------------------------------------- the register itself


def test_register_validation(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    with pytest.raises(claims.ClaimError, match="already registered"):
        _register(bench, "C1", "Q1", "effect")
    with pytest.raises(claims.ClaimError, match="prediction"):
        _register(bench, "C2", "Q1", "better")
    with pytest.raises(claims.ClaimError, match="not in"):
        _register(bench, "C3", "Q9", "effect")
    with pytest.raises(claims.ClaimError, match="claim id"):
        _register(bench, "C 4", "Q1", "effect")


def test_edited_register_is_refused(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    _register(bench, "C2", "Q2", "no_effect")
    log = _paths(bench)["log"]
    lines = log.read_text().splitlines()
    first = json.loads(lines[0])
    first["prediction"] = "no_effect"
    lines[0] = json.dumps(first, sort_keys=True)
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(LogError, match="hash chain"):
        claims.load_claims(log)


def test_document_check(bench):
    _setup(bench)
    _register(bench, "C1", "Q1", "effect")
    _register(bench, "C2", "Q2", "no_effect")
    p = _paths(bench)
    doc = bench.root / "report.md"
    doc.write_text("TF-IDF leads on Phase 1 [claim:C1]. Logreg does not dominate [claim:C2].\n")
    with pytest.raises(claims.ClaimError, match="no result yet"):
        claims.check_document(p["log"], doc)
    _run_and_decide(bench)
    claims.resolve(p["log"], verdicts=p["verdicts"])
    assert set(claims.check_document(p["log"], doc)) == {"C1", "C2"}
    doc.write_text("An unregistered finding [claim:C7].\n")
    with pytest.raises(claims.ClaimError, match="C7 is cited but not in the register"):
        claims.check_document(p["log"], doc)


def test_cli_claims(bench, capsys):
    _setup(bench)
    r = str(bench.root)
    base = ["claims", "--log", f"{r}/claims.jsonl"]
    assert (
        main(
            [
                *base,
                "register",
                "C1",
                "--statement",
                "TF-IDF leads",
                "--question",
                "Q1",
                "--prediction",
                "effect",
                "--questions",
                f"{r}/questions.yaml",
                "--construct",
                f"{r}/construct.yaml",
            ]
        )
        == 0
    )
    _run_and_decide(bench)
    assert main([*base, "resolve", "--verdicts", f"{r}/verdicts.json"]) == 0
    assert main([*base, "show"]) == 0
    out = capsys.readouterr().out
    assert "C1" in out and "registered 0" in out
    (bench.root / "doc.md").write_text("[claim:C1]")
    assert main([*base, "check-doc", f"{r}/doc.md"]) == 0
    (bench.root / "doc.md").write_text("[claim:C2]")
    assert main([*base, "check-doc", f"{r}/doc.md"]) == 1
