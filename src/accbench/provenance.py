"""X1 provenance: stamp on write, check on read, refuse a mismatch.

Every artifact accbench writes carries a ``_provenance`` block with:

- the **code state** it was produced under: a sha256 over the source files in the
  working tree, so uncommitted changes count. The git commit and a dirty flag are
  recorded beside it for reference only and are never trusted on their own;
- the sha256 of every **input** it was computed from.

A step that reads an artifact checks both. If an input has changed since the
artifact was written, the artifact is stale and the read is refused. If the code
state differs from the one the reader expects, the read is refused unless the caller
passes ``allow_code_drift=True``, and that override is carried into whatever the
caller writes next.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from accbench import __version__
from accbench.errors import CodeStateMismatch, MissingProvenance, StaleInput

STAMP_KEY = "_provenance"

DEFAULT_CODE_PATTERNS = ("*.py", "pyproject.toml", "requirements*.txt")
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "build",
        "dist",
    }
)
_PACKAGE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------- hashing


def file_digest(path: str | os.PathLike[str]) -> str:
    """sha256 of a file's exact bytes, or of a directory's files and their relative paths."""
    p = Path(path)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(q for q in p.rglob("*") if q.is_file()):
            h.update(f.relative_to(p).as_posix().encode())
            h.update(b"\0")
            h.update(file_digest(f).encode())
            h.update(b"\n")
        return h.hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_bytes(path: Path) -> bytes:
    # Line endings are normalized so a Windows and a Linux checkout of the same
    # code have the same code state.
    return path.read_bytes().replace(b"\r\n", b"\n")


def _walk_sources(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            if any(fnmatch(name, pat) for pat in patterns):
                yield Path(dirpath) / name


# ---------------------------------------------------------------- code state


@dataclass(frozen=True)
class CodeState:
    """The state of the code that produced an artifact.

    Two code states match when their ``digest`` matches. ``git_sha`` and ``git_dirty``
    are recorded for a reader's convenience and play no part in the comparison.
    """

    digest: str
    files: int
    git_sha: str | None = None
    git_dirty: bool | None = None
    accbench_version: str = __version__
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodeState:
        return cls(
            digest=data["digest"],
            files=data["files"],
            git_sha=data.get("git_sha"),
            git_dirty=data.get("git_dirty"),
            accbench_version=data.get("accbench_version", ""),
            environment=dict(data.get("environment", {})),
        )

    def short(self) -> str:
        return self.digest[:12]


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            continue
    return env


def code_state(
    roots: Sequence[str | os.PathLike[str]] | None = None,
    *,
    patterns: Sequence[str] = DEFAULT_CODE_PATTERNS,
    include_accbench: bool = True,
) -> CodeState:
    """Hash the source files under ``roots`` (default: the current directory).

    accbench's own source is included by default, because a change to the library
    changes results as surely as a change to the caller's code.
    """
    root_paths = [Path(r).resolve() for r in (roots or [Path.cwd()])]
    if include_accbench and _PACKAGE_DIR not in root_paths:
        root_paths.append(_PACKAGE_DIR)

    h = hashlib.sha256()
    count = 0
    seen: set[Path] = set()
    for i, root in enumerate(root_paths):
        for f in _walk_sources(root, patterns):
            if f in seen:
                continue
            seen.add(f)
            label = f"{i}:{f.relative_to(root).as_posix()}"
            h.update(label.encode())
            h.update(b"\0")
            h.update(hashlib.sha256(_source_bytes(f)).hexdigest().encode())
            h.update(b"\n")
            count += 1

    first = root_paths[0]
    git_sha = _git(first, "rev-parse", "HEAD")
    git_dirty = None
    if git_sha is not None:
        status = _git(first, "status", "--porcelain")
        git_dirty = None if status is None else bool(status)

    return CodeState(
        digest=h.hexdigest(),
        files=count,
        git_sha=git_sha,
        git_dirty=git_dirty,
        environment=_environment(),
    )


# ---------------------------------------------------------------- stamping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def stamp(
    payload: Mapping[str, Any],
    *,
    code: CodeState,
    inputs: Mapping[str, str | os.PathLike[str]] | None = None,
    overrides: Sequence[str] = (),
    base: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return ``payload`` with a provenance block recording ``code`` and each input's sha256.

    Input paths are stored relative to ``base`` (the directory the artifact will be
    written to) when one is given, so an output tree can be moved as a whole.
    """
    if STAMP_KEY in payload:
        raise ValueError(f"payload already carries a {STAMP_KEY!r} block")
    recorded = {}
    for name, p in (inputs or {}).items():
        path = Path(p)
        if not path.exists():
            raise StaleInput(f"input {name!r} does not exist: {path}")
        recorded[name] = {"path": _store_path(path, base), "sha256": file_digest(path)}
    block: dict[str, Any] = {
        "written_at": _now(),
        "code_state": code.to_dict(),
        "inputs": recorded,
    }
    if overrides:
        block["overrides"] = list(overrides)
    return {**payload, STAMP_KEY: block}


def _store_path(path: Path, base: str | os.PathLike[str] | None) -> str:
    resolved = path.resolve()
    if base is None:
        return resolved.as_posix()
    try:
        return Path(os.path.relpath(resolved, Path(base).resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return resolved.as_posix()


def _resolve_input(artifact_path: Path, stored: str) -> Path:
    q = Path(stored)
    return q if q.is_absolute() else artifact_path.parent / q


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the new one, never half."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_artifact(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    code: CodeState,
    inputs: Mapping[str, str | os.PathLike[str]] | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Stamp ``payload`` and write it atomically as JSON. Returns the stamped payload."""
    stamped = stamp(payload, code=code, inputs=inputs, overrides=overrides, base=Path(path).parent)
    atomic_write_text(path, dumps(stamped))
    return stamped


# ---------------------------------------------------------------- checking


@dataclass(frozen=True)
class Artifact:
    """A checked artifact: its payload, where it came from, and any override taken on read."""

    path: Path
    payload: dict[str, Any]
    code: CodeState
    inputs: dict[str, dict[str, str]]
    overrides: tuple[str, ...] = ()

    @property
    def data(self) -> dict[str, Any]:
        return {k: v for k, v in self.payload.items() if k != STAMP_KEY}


def check_artifact(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    expected_code: CodeState | None = None,
    allow_code_drift: bool = False,
) -> Artifact:
    """Check a loaded payload's stamp. See :func:`read_artifact`."""
    p = Path(path)
    block = payload.get(STAMP_KEY)
    if not isinstance(block, Mapping) or "code_state" not in block:
        raise MissingProvenance(f"{p} carries no provenance stamp; it cannot be checked")

    recorded_code = CodeState.from_dict(block["code_state"])
    inputs = {k: dict(v) for k, v in block.get("inputs", {}).items()}

    stale = []
    for name, rec in inputs.items():
        ip = _resolve_input(p, rec["path"])
        if not ip.exists():
            stale.append(f"{name!r} ({ip}) no longer exists")
            continue
        now = file_digest(ip)
        if now != rec["sha256"]:
            stale.append(f"{name!r} ({ip}) changed: recorded {rec['sha256'][:12]}, now {now[:12]}")
    if stale:
        raise StaleInput(f"{p} is stale, recompute it: " + "; ".join(stale))

    overrides: list[str] = list(block.get("overrides", []))
    if expected_code is not None and recorded_code.digest != expected_code.digest:
        msg = (
            f"{p} was written under code state {recorded_code.short()}, "
            f"expected {expected_code.short()}"
        )
        if not allow_code_drift:
            raise CodeStateMismatch(msg + "; recompute it, or pass allow_code_drift to override")
        overrides.append(f"code drift accepted on read: {msg}")

    return Artifact(
        path=p,
        payload=dict(payload),
        code=recorded_code,
        inputs=inputs,
        overrides=tuple(overrides),
    )


def read_artifact(
    path: str | os.PathLike[str],
    *,
    expected_code: CodeState | None = None,
    allow_code_drift: bool = False,
) -> Artifact:
    """Load a JSON artifact and check its stamp.

    Refuses (raises) when the artifact has no stamp, when any recorded input has
    changed or disappeared, or when ``expected_code`` is given and differs from the
    recorded code state. With ``allow_code_drift`` the last check becomes an override
    recorded on the returned :class:`Artifact`; pass ``artifact.overrides`` on to the
    next :func:`write_artifact` so the override travels with the result.
    """
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return check_artifact(
        p, payload, expected_code=expected_code, allow_code_drift=allow_code_drift
    )


def require_same_code(artifacts: Sequence[Artifact]) -> CodeState:
    """Refuse a set of artifacts that were not all produced under one code state."""
    if not artifacts:
        raise ValueError("no artifacts given")
    states: dict[str, list[Path]] = {}
    for a in artifacts:
        states.setdefault(a.code.digest, []).append(a.path)
    if len(states) > 1:
        parts = [
            f"{digest[:12]}: {len(paths)} artifact(s), e.g. {paths[0]}"
            for digest, paths in states.items()
        ]
        raise CodeStateMismatch(
            "artifacts were produced under different code states: " + "; ".join(parts)
        )
    return artifacts[0].code
