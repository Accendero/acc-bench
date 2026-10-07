"""Unit 0, construct (R0 meaning): every reported number stands in for a named decision.

The construct statement (``construct.yaml``) states the reader's question, the decision
it informs, the target, the metric and the tasks, each with its reason, before anything
is frozen. Where two measures of one quantity exist, it names the one used for claims
and why. The fixture builder refuses to freeze without a statement that passes
:func:`validate`.

Schema (every key below is required unless marked optional; unknown keys are errors,
because a misspelt key is silently ignored otherwise)::

    version: 1                  # bumped by every amendment
    stated_on: 2026-10-07       # the date the statement was written
    owner: "A. Person"          # who is accountable for it (X2)
    question: "..."             # the reader's question
    decision: "..."             # the decision the answer informs
    target: {name: ..., definition: ...}
    metric:
      name: pr_auc              # one of accbench.metrics.METRICS
      reason: "..."
      alternatives:             # optional
        - {name: roc_auc, why_not: "..."}
    tasks:
      - {name: mortality, reason: "..."}
    reported_numbers:           # each number the benchmark reports, and its decision
      - {name: pr_auc_by_cell, decision: "..."}
    measures:                   # optional
      - {quantity: ..., used: ..., instead_of: ..., reason: ...}
    amendments:                 # required once version > 1, one per version from 2
      - {version: 2, date: 2026-11-01, reason: "..."}
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from accbench.errors import ConstructError
from accbench.metrics import METRICS
from accbench.provenance import file_digest

TOP_LEVEL = {
    "version": "required",
    "stated_on": "required",
    "owner": "required",
    "question": "required",
    "decision": "required",
    "target": "required",
    "metric": "required",
    "tasks": "required",
    "reported_numbers": "required",
    "measures": "optional",
    "amendments": "optional",
}

_ENTRY_FIELDS = {
    "target": ({"name", "definition"}, set()),
    "metric": ({"name", "reason"}, {"alternatives"}),
    "alternatives": ({"name", "why_not"}, set()),
    "tasks": ({"name", "reason"}, set()),
    "reported_numbers": ({"name", "decision"}, set()),
    "measures": ({"quantity", "used", "instead_of", "reason"}, set()),
    "amendments": ({"version", "date", "reason"}, set()),
}


@dataclass(frozen=True)
class Construct:
    """A construct statement that has passed its schema check."""

    path: Path
    sha256: str
    data: dict[str, Any]

    @property
    def version(self) -> int:
        return int(self.data["version"])

    @property
    def metric(self) -> str:
        return str(self.data["metric"]["name"])

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(str(t["name"]) for t in self.data["tasks"])

    def summary(self) -> str:
        return (
            f"version {self.version}, metric {self.metric}, "
            f"{len(self.tasks)} task(s): {', '.join(self.tasks)}; sha256 {self.sha256[:12]}"
        )


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _as_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _check_entry(where: str, kind: str, entry: Any, problems: list[str]) -> None:
    required, optional = _ENTRY_FIELDS[kind]
    if not isinstance(entry, Mapping):
        problems.append(f"{where}: must be a mapping with keys {sorted(required)}")
        return
    for key in sorted(required):
        if key not in entry:
            problems.append(f"{where}.{key}: missing")
        elif key not in ("version", "date") and not _nonempty_str(entry[key]):
            problems.append(f"{where}.{key}: must be a non-empty string")
    for key in sorted(set(entry) - required - optional):
        problems.append(f"{where}.{key}: unknown key")


def _check_list(data: Mapping[str, Any], kind: str, problems: list[str], *, min_len: int) -> list:
    items = data.get(kind)
    if items is None:
        return []
    if not isinstance(items, list):
        problems.append(f"{kind}: must be a list")
        return []
    if len(items) < min_len:
        problems.append(f"{kind}: must have at least {min_len} entry")
        return []
    for i, entry in enumerate(items):
        _check_entry(f"{kind}[{i}]", kind, entry, problems)
    return items


def validate(data: Any) -> list[str]:
    """Return every problem with a construct statement; an empty list means it passes."""
    if not isinstance(data, Mapping):
        return ["the statement must be a YAML mapping"]
    problems: list[str] = []

    for key, need in TOP_LEVEL.items():
        if need == "required" and key not in data:
            problems.append(f"{key}: missing")
    for key in sorted(set(data) - set(TOP_LEVEL)):
        problems.append(f"{key}: unknown key")

    version = data.get("version")
    if "version" in data and (
        not isinstance(version, int) or isinstance(version, bool) or version < 1
    ):
        problems.append("version: must be an integer of at least 1")
        version = None
    if "stated_on" in data and _as_date(data["stated_on"]) is None:
        problems.append("stated_on: must be a date (YYYY-MM-DD)")
    for key in ("owner", "question", "decision"):
        if key in data and not _nonempty_str(data[key]):
            problems.append(f"{key}: must be a non-empty string")

    if "target" in data:
        _check_entry("target", "target", data["target"], problems)

    metric = data.get("metric")
    if "metric" in data:
        _check_entry("metric", "metric", metric, problems)
        if isinstance(metric, Mapping):
            name = metric.get("name")
            if _nonempty_str(name) and name not in METRICS:
                problems.append(f"metric.name: {name!r} is not one of {sorted(METRICS)}")
            alts = metric.get("alternatives")
            if alts is not None:
                if not isinstance(alts, list):
                    problems.append("metric.alternatives: must be a list")
                else:
                    for i, alt in enumerate(alts):
                        _check_entry(f"metric.alternatives[{i}]", "alternatives", alt, problems)

    tasks = _check_list(data, "tasks", problems, min_len=1)
    names = [t.get("name") for t in tasks if isinstance(t, Mapping)]
    for dup in sorted({n for n in names if isinstance(n, str) and names.count(n) > 1}):
        problems.append(f"tasks: {dup!r} is listed more than once")

    _check_list(data, "reported_numbers", problems, min_len=1)
    _check_list(data, "measures", problems, min_len=0)
    amendments = _check_list(data, "amendments", problems, min_len=0)

    if isinstance(version, int):
        seen = []
        for i, a in enumerate(amendments):
            if not isinstance(a, Mapping):
                continue
            v = a.get("version")
            if not isinstance(v, int) or isinstance(v, bool):
                problems.append(f"amendments[{i}].version: must be an integer")
            else:
                seen.append(v)
            if "date" in a and _as_date(a.get("date")) is None:
                problems.append(f"amendments[{i}].date: must be a date (YYYY-MM-DD)")
        expected = list(range(2, version + 1))
        if seen != expected:
            problems.append(
                f"amendments: version {version} needs one amendment per version "
                f"{expected or '(none)'} in order, found {seen or '(none)'}"
            )

    return problems


def load_construct(path: str | os.PathLike[str]) -> Construct:
    """Load and check a construct statement; raise :class:`ConstructError` listing every problem."""
    p = Path(path)
    if not p.is_file():
        raise ConstructError(p, ["file not found; state the construct before freezing anything"])
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConstructError(p, [f"not valid YAML: {exc}"]) from exc
    problems = validate(data)
    if problems:
        raise ConstructError(p, problems)
    return Construct(path=p, sha256=file_digest(p), data=dict(data))


require_construct = load_construct
"""Alias used by later units: the guard that blocks a fixture freeze."""
