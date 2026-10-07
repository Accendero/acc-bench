"""The repair log: every fix to the harness, with its measured effect.

Section 8 asks that each repair run as a measured ablation, so the size of every defect
is a number. :func:`record_repair` refuses an entry without a before and an after on
the same cells, and appends it to ``repairs.jsonl``, an append-only hash-chained log.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from accbench.appendlog import append, read_log
from accbench.errors import LogError


def record_repair(
    log: str | os.PathLike[str],
    *,
    description: str,
    columns: Sequence[str],
    metric: str,
    before: Mapping[str, float],
    after: Mapping[str, float],
    evidence: Sequence[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Append one measured repair. ``before`` and ``after`` map each cell to its score."""
    problems = []
    if not description.strip():
        problems.append("describe the repair")
    if not columns:
        problems.append("name the columns the repair touched")
    if not before or not after:
        problems.append("a repair needs a measured before and after")
    elif set(before) != set(after):
        problems.append(
            f"before and after must cover the same cells (only before: "
            f"{sorted(set(before) - set(after))}, only after: {sorted(set(after) - set(before))})"
        )
    if problems:
        raise LogError("repair refused: " + "; ".join(problems))
    delta = {cell: float(after[cell]) - float(before[cell]) for cell in sorted(before)}
    return append(
        log,
        {
            "kind": "repair",
            "description": description,
            "columns": list(columns),
            "metric": metric,
            "before": {k: float(v) for k, v in sorted(before.items())},
            "after": {k: float(v) for k, v in sorted(after.items())},
            "delta": delta,
            "mean_delta": sum(delta.values()) / len(delta),
            "evidence": [Path(e).as_posix() for e in evidence],
        },
    )


def read_repairs(log: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return read_log(log)
