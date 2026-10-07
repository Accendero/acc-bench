"""Append-only JSON-lines logs with a hash chain.

Each line carries the sha256 of the line before it (``prev``) and of itself (``hash``),
so an edited, reordered or deleted line breaks the chain and :func:`read_log` refuses
the file. Used by the repair log (unit 2) and the claims registry (unit 6).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accbench.errors import LogError

GENESIS = "0" * 64


def _entry_hash(entry: Mapping[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def read_log(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a log and verify its chain. A missing file is an empty log."""
    p = Path(path)
    if not p.exists():
        return []
    entries: list[dict[str, Any]] = []
    prev = GENESIS
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogError(f"{p} line {n} is not valid JSON") from exc
        if entry.get("prev") != prev or entry.get("hash") != _entry_hash(entry):
            raise LogError(f"{p} line {n} breaks the hash chain; the log was edited")
        prev = entry["hash"]
        entries.append(entry)
    return entries


def append(path: str | os.PathLike[str], record: Mapping[str, Any]) -> dict[str, Any]:
    """Append ``record`` to the log, after verifying the chain so far. Returns the entry."""
    reserved = {"prev", "hash", "seq", "recorded_at"} & set(record)
    if reserved:
        raise LogError(f"record may not set {sorted(reserved)}")
    p = Path(path)
    existing = read_log(p)
    entry = {
        **record,
        "seq": len(existing) + 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "prev": existing[-1]["hash"] if existing else GENESIS,
    }
    entry["hash"] = _entry_hash(entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return entry
