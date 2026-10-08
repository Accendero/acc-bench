"""Unit 6, claims (R6 accountable claims): registered before results, amended in the open.

The claims register (``claims.jsonl``) is an append-only, hash-chained log. Four kinds of
entry, each dated when it is written:

- ``register``: a claim, the pre-registered question (the test) that will decide it, the
  predicted outcome, and a snapshot of the question's rule and parameters;
- ``amend``: a change to the prediction or the question, with its reason. An amendment
  is made in the open: it requires the construct statement to have moved to a new
  version (with its own dated amendment) since the claim last changed;
- ``resolve``: the verdict that decided the claim, and the claim's status;
- ``untested``: a claim that will not be tested, with the reason.

Statuses: ``registered``, ``confirmed``, ``refuted``, ``inverted`` (an effect in the
opposite direction), ``inconclusive`` and ``untested``. Refuted and untested rows stay in
the register with the same weight as confirmed ones.

:func:`resolve` refuses a claim registered after the earliest run record behind its
verdict, a verdict whose question no longer matches the registered snapshot, and an
oracle verdict for a confirmatory claim. :func:`check_document` refuses a document that
cites a claim the register does not hold, or one that has no result.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from accbench.appendlog import append, read_log
from accbench.construct import load_construct
from accbench.errors import AccbenchError
from accbench.provenance import code_state, file_digest, read_artifact
from accbench.rules import check_questions
from accbench.runner import history_path

STATUSES = ("registered", "confirmed", "refuted", "inverted", "inconclusive", "untested")
DOC_PATTERN = r"\[claim:([A-Za-z0-9_.\-]+)\]"


class ClaimError(AccbenchError):
    """A claim entry was refused."""


@dataclass
class Claim:
    id: str
    statement: str
    question: str
    prediction: str
    kind: str
    snapshot: dict[str, Any]
    construct_version: int
    registered_at: str
    status: str = "registered"
    amendments: list[dict[str, Any]] = field(default_factory=list)
    resolution: dict[str, Any] | None = None
    untested_reason: str | None = None

    @property
    def last_changed_at(self) -> str:
        return self.amendments[-1]["recorded_at"] if self.amendments else self.registered_at


def _parse_prediction(prediction: str) -> tuple[str, str | None]:
    outcome, _, direction = prediction.partition(":")
    if outcome not in ("effect", "no_effect") or (outcome == "no_effect" and direction):
        raise ClaimError(
            "prediction must be 'effect', 'effect:a', 'effect:b', 'effect:<the better method>' "
            f"or 'no_effect'; got {prediction!r}"
        )
    return outcome, direction or None


def _check_prediction(prediction: str, snapshot: Mapping[str, Any]) -> None:
    """Refuse a prediction its question's rule cannot decide, when it is written."""
    from accbench.rules import _methods_in, get_rule

    _, direction = _parse_prediction(prediction)
    if direction is None:
        return
    rule = snapshot["rule"]
    if not get_rule(rule).directional:
        raise ClaimError(
            f"rule {rule} does not report which method is better; predict 'effect' or 'no_effect'"
        )
    params = snapshot["params"]
    named = {m for side in ("a", "b") for m in _methods_in(params.get(side))}
    if direction not in ("a", "b") and direction not in named:
        raise ClaimError(
            f"the prediction names {direction!r}, which is not a or b and not a method of this "
            f"question ({sorted(named)})"
        )


def load_claims(log: str | os.PathLike[str]) -> dict[str, Claim]:
    """Replay the log into the current state of every claim."""
    claims: dict[str, Claim] = {}
    for e in read_log(log):
        cid = e["claim_id"]
        if e["kind"] == "register":
            claims[cid] = Claim(
                id=cid,
                statement=e["statement"],
                question=e["question"],
                prediction=e["prediction"],
                kind=e["claim_kind"],
                snapshot=e["snapshot"],
                construct_version=e["construct_version"],
                registered_at=e["recorded_at"],
            )
            continue
        c = claims[cid]
        if e["kind"] == "amend":
            c.amendments.append(e)
            c.statement = e.get("statement", c.statement)
            c.prediction = e.get("prediction", c.prediction)
            c.question = e.get("question", c.question)
            c.snapshot = e.get("snapshot", c.snapshot)
            c.construct_version = e["construct_version"]
            c.status, c.resolution = "registered", None
        elif e["kind"] == "resolve":
            c.status, c.resolution = e["status"], e
        elif e["kind"] == "untested":
            c.status, c.untested_reason = "untested", e["reason"]
    return claims


def _snapshot(questions_path: Path, question_id: str) -> dict[str, Any]:
    q = check_questions(questions_path)
    for question in q["questions"]:
        if question["id"] == question_id:
            return {"rule": question["rule"], "params": question["params"]}
    raise ClaimError(f"question {question_id!r} is not in {questions_path}")


def register(
    log: str | os.PathLike[str],
    *,
    claim_id: str,
    statement: str,
    question: str,
    prediction: str,
    questions: str | os.PathLike[str],
    construct: str | os.PathLike[str],
    exploratory: bool = False,
) -> dict[str, Any]:
    """Register a claim and the question that will decide it, before any result exists."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", claim_id):
        raise ClaimError(f"claim id {claim_id!r} may use letters, digits, '.', '_' and '-'")
    if claim_id in load_claims(log):
        raise ClaimError(f"claim {claim_id!r} is already registered; amend it instead")
    if not statement.strip():
        raise ClaimError("state the claim")
    _parse_prediction(prediction)
    c = load_construct(construct)
    snapshot = _snapshot(Path(questions), question)
    _check_prediction(prediction, snapshot)
    return append(
        log,
        {
            "kind": "register",
            "claim_id": claim_id,
            "statement": statement,
            "question": question,
            "prediction": prediction,
            "claim_kind": "exploratory" if exploratory else "confirmatory",
            "snapshot": snapshot,
            "construct_version": c.version,
            "construct_sha256": c.sha256,
        },
    )


def amend(
    log: str | os.PathLike[str],
    *,
    claim_id: str,
    reason: str,
    construct: str | os.PathLike[str],
    prediction: str | None = None,
    question: str | None = None,
    questions: str | os.PathLike[str] | None = None,
    statement: str | None = None,
) -> dict[str, Any]:
    """Amend a claim in the open. The construct must have a newer version than the claim's."""
    claims = load_claims(log)
    if claim_id not in claims:
        raise ClaimError(f"claim {claim_id!r} is not registered")
    if not reason.strip():
        raise ClaimError("an amendment needs its reason")
    if prediction is None and question is None and statement is None:
        raise ClaimError("an amendment changes the statement, the prediction or the question")
    claim = claims[claim_id]
    c = load_construct(construct)
    if c.version <= claim.construct_version:
        raise ClaimError(
            f"amending {claim_id} needs the construct statement amended first: it is at version "
            f"{c.version}, the claim was made under version {claim.construct_version}"
        )
    record: dict[str, Any] = {
        "kind": "amend",
        "claim_id": claim_id,
        "reason": reason,
        "construct_version": c.version,
        "construct_sha256": c.sha256,
        "had_result": claim.resolution is not None,
    }
    if statement is not None:
        if not statement.strip():
            raise ClaimError("state the claim")
        record["statement"] = statement
    if prediction is not None:
        _parse_prediction(prediction)
        record["prediction"] = prediction
    if question is not None or prediction is not None:
        if questions is None:
            raise ClaimError("pass the questions file so the amended question can be snapshotted")
        record["question"] = question or claim.question
        record["snapshot"] = _snapshot(Path(questions), record["question"])
        _check_prediction(prediction or claim.prediction, record["snapshot"])
    return append(log, record)


def mark_untested(log: str | os.PathLike[str], *, claim_id: str, reason: str) -> dict[str, Any]:
    if claim_id not in load_claims(log):
        raise ClaimError(f"claim {claim_id!r} is not registered")
    if not reason.strip():
        raise ClaimError("say why the claim will not be tested")
    return append(log, {"kind": "untested", "claim_id": claim_id, "reason": reason})


def _earliest_run(runs_dir: Path) -> str | None:
    """The time the first result behind these runs existed.

    The runner's history log keeps every record it ever wrote, so a re-run, which
    replaces the records, cannot move this time later.
    """
    times = [e["recorded_at"] for e in read_log(history_path(runs_dir))]
    for p in runs_dir.rglob("*.json"):
        try:
            stamp = json.loads(p.read_text(encoding="utf-8")).get("_provenance", {})
        except (OSError, json.JSONDecodeError):
            continue
        if "written_at" in stamp:
            times.append(stamp["written_at"])
    return min(times, key=datetime.fromisoformat) if times else None


def status_for(prediction: str, verdict: Mapping[str, Any]) -> str:
    want, direction = _parse_prediction(prediction)
    outcome = verdict["outcome"]
    if outcome == "inconclusive":
        return "inconclusive"
    if want == "no_effect":
        return "confirmed" if outcome == "no_effect" else "refuted"
    if outcome == "no_effect":
        return "refuted"
    if direction is None:
        return "confirmed"
    details = verdict.get("details", {})
    better = details.get("better")
    if direction in ("a", "b"):
        direction = details.get(direction, direction)
    if better is None:
        raise ClaimError(
            f"the prediction names a direction ({direction}) but rule {verdict['rule']!r} "
            "reports none; register the claim as 'effect'"
        )
    return "confirmed" if better == direction else "inverted"


def resolve(
    log: str | os.PathLike[str], *, verdicts: str | os.PathLike[str]
) -> list[dict[str, Any]]:
    """Resolve every registered claim whose question the verdicts answer."""
    vpath = Path(verdicts)
    art = read_artifact(vpath)  # refuses stale verdicts
    by_question = {v["id"]: v for v in art.data["verdicts"]}
    runs_rec = art.inputs.get("runs")
    if runs_rec is None:
        raise ClaimError(f"{vpath} does not record the runs behind it")
    runs_dir = Path(runs_rec["path"])
    runs_dir = runs_dir if runs_dir.is_absolute() else vpath.parent / runs_dir
    earliest = _earliest_run(runs_dir)

    entries, problems = [], []
    claims = load_claims(log)
    for claim in claims.values():
        v = by_question.get(claim.question)
        if v is None or claim.status == "untested":
            continue
        if earliest and datetime.fromisoformat(claim.registered_at) >= datetime.fromisoformat(
            earliest
        ):
            problems.append(
                f"{claim.id} was registered at {claim.registered_at}, after the first run "
                f"({earliest}); a claim written after its result cannot be resolved as registered"
            )
            continue
        if {"rule": v["rule"], "params": v["params"]} != claim.snapshot:
            problems.append(
                f"{claim.id}: question {claim.question} has changed since the claim was registered "
                "or last amended; amend the claim in the open first"
            )
            continue
        if v["oracle"] and claim.kind == "confirmatory":
            problems.append(
                f"{claim.id}: the verdict for {claim.question} used an oracle selection on test "
                "data, which cannot decide a confirmatory claim"
            )
            continue
        status = status_for(claim.prediction, v)
        if (
            claim.status == status
            and claim.resolution
            and claim.resolution.get("verdicts_sha256") == _sha(vpath)
        ):
            continue  # already resolved by these verdicts
        entries.append(
            {
                "kind": "resolve",
                "claim_id": claim.id,
                "question": claim.question,
                "rule": v["rule"],
                "outcome": v["outcome"],
                "summary": v["summary"],
                "status": status,
                "verdicts_path": vpath.as_posix(),
                "verdicts_sha256": _sha(vpath),
                "amended_after_results": any(
                    datetime.fromisoformat(a["recorded_at"]) >= datetime.fromisoformat(earliest)
                    for a in claims[claim.id].amendments
                )
                if earliest
                else False,
            }
        )
    if problems:
        raise ClaimError("claims refused:\n" + "\n".join(f"  - {p}" for p in problems))
    return [append(log, e) for e in entries]


def _sha(path: Path) -> str:
    return file_digest(path)


def _result_is_current(claim: Claim, log: Path) -> str | None:
    """Why a claim's recorded result can no longer be used, or None if it can."""
    if claim.resolution is None:
        return None
    vpath = Path(claim.resolution["verdicts_path"])
    if not vpath.is_absolute() and not vpath.exists():
        vpath = log.parent / vpath
    if not vpath.exists():
        return f"its verdicts file {vpath} is gone"
    if file_digest(vpath) != claim.resolution["verdicts_sha256"]:
        return "the verdicts changed after it was resolved; run acc-bench claims resolve"
    try:
        read_artifact(vpath, expected_code=code_state([log.resolve().parent]))
    except AccbenchError as exc:
        return f"its verdicts are out of date ({exc})"
    return None


def check_document(
    log: str | os.PathLike[str],
    document: str | os.PathLike[str],
    *,
    pattern: str = DOC_PATTERN,
    require_all: bool = False,
) -> dict[str, Any]:
    """Every claim a document cites must be in the register, with a current result.

    Claims are cited as ``[claim:C1]`` by default. A result is current when the verdicts
    behind it are unchanged and were written by the code running now. With
    ``require_all``, every claim in the register must be cited. The check does not read
    the text around a citation; a person must make sure the text agrees with the result.
    Returns the status of each cited claim and the claims the document does not cite.
    """
    text = Path(document).read_text(encoding="utf-8")
    cited = sorted(set(re.findall(pattern, text)))
    claims = load_claims(log)
    problems = []
    for cid in cited:
        if cid not in claims:
            problems.append(f"{cid} is cited but not in the register")
        elif claims[cid].status == "registered":
            problems.append(f"{cid} is cited but has no result yet")
        elif why := _result_is_current(claims[cid], Path(log)):
            problems.append(f"{cid}: {why}")
    uncited = sorted(set(claims) - set(cited))
    if require_all and uncited and not problems:
        raise ClaimError(f"{document} does not cite these registered claims: {uncited}")
    if require_all and uncited:
        problems.append(f"the document does not cite these registered claims: {uncited}")
    if problems:
        raise ClaimError(
            f"{document} cites claims the register cannot back: " + "; ".join(problems)
        )
    return {"cited": {cid: claims[cid].status for cid in cited}, "uncited": uncited}


def table(log: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The register as rows: one per claim, with its test, status and dates."""
    rows = []
    for c in load_claims(log).values():
        rows.append(
            {
                "claim": c.id,
                "statement": c.statement,
                "test": c.question,
                "rule": c.snapshot["rule"],
                "prediction": c.prediction,
                "kind": c.kind,
                "status": c.status,
                "registered_at": c.registered_at,
                "amendments": [
                    {"at": a["recorded_at"], "reason": a["reason"]} for a in c.amendments
                ],
                "resolved_at": c.resolution["recorded_at"] if c.resolution else None,
                "amended_after_results": bool(
                    c.resolution and c.resolution.get("amended_after_results")
                ),
                "evidence": c.resolution["verdicts_path"] if c.resolution else None,
                "untested_reason": c.untested_reason,
            }
        )
    return rows


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out = dict.fromkeys(STATUSES, 0)
    for r in rows:
        out[r["status"]] += 1
    return out
