"""Unit 3, runner (R3 completeness): every planned run leaves a record or a recorded reason.

The grid (``grid.yaml``) names the fixture, the channel registry and manifest, the
methods, the split keys and fit seeds, the thread count and, optionally, a price table::

    construct: construct.yaml
    fixture: fixtures.yaml           # the spec (or a frozen fixtures/<hash> directory)
    fixtures_dir: fixtures           # optional, where frozen fixtures live (default fixtures)
    registry: channels.yaml
    channel_manifest: channel_manifest.json
    methods: [majority, logreg, hist_gbm, tfidf_logreg]
    methods_module: my_methods.py    # optional: registers extra methods
    tasks: [mortality]               # optional, default every task in the fixture
    split_keys: [split-0]            # optional, default the fixture's primary key
    fit_seeds: [0]                   # optional
    threads: 1                       # optional
    prices: prices.yaml              # optional; without it a billable call fails closed
    out: runs

One run is (cell, method, split key, fit seed). Each writes ``<out>/<task>/<run>.json``
atomically, with its predictions beside it, so an interrupted grid resumes where it
stopped. A method that cannot run here leaves a ``skipped`` record with its reason; a
method that fails leaves an ``error`` record with the traceback, and the grid carries
on. Every record is stamped with the working-tree code state, the thread count, its
cost against the pinned price table, and the hashes of its inputs.

On resume, a record written under different code, or over inputs that have changed
since, is refused unless ``on_stale="rerun"``: a grid never mixes code states.

:func:`coverage` counts ok, skipped, error, missing and stale runs against the plan,
and :func:`require_complete` refuses a grid with any run that is neither ok nor a
recorded skip. That refusal is the guard on the campaign's leaderboard, which averaged
each method over whichever cells it had finished.
"""

from __future__ import annotations

import importlib.util
import os
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from accbench import metrics
from accbench.channels import Featurizer, load_registry
from accbench.construct import load_construct
from accbench.costs import Meter, load_prices
from accbench.errors import AccbenchError, ProvenanceError
from accbench.fixtures import load_fixture, locate_fixture
from accbench.methods import available_methods, make_method
from accbench.provenance import (
    CodeState,
    atomic_write_text,
    code_state,
    read_artifact,
    write_artifact,
)

GRID_KEYS = {
    "construct",
    "fixture",
    "fixtures_dir",
    "registry",
    "channel_manifest",
    "methods",
    "methods_module",
    "tasks",
    "split_keys",
    "fit_seeds",
    "threads",
    "prices",
    "out",
}
REQUIRED = ("construct", "fixture", "registry", "channel_manifest", "methods", "out")
STATUSES = ("ok", "skipped", "error", "missing", "stale")


class GridError(AccbenchError):
    """The grid is invalid, or a run cannot be planned or resumed."""


class IncompleteGrid(GridError):
    """A grid has runs that are neither complete nor recorded skips."""


@dataclass(frozen=True)
class RunKey:
    task: str
    cell: str
    method: str
    split_key: str
    fit_seed: int

    @property
    def id(self) -> str:
        cell = self.cell.replace("/", "--")
        return f"{cell}__{self.method}__{self.split_key}__s{self.fit_seed}"

    def record_path(self, out: Path) -> Path:
        return out / self.task / f"{self.id}.json"

    def predictions_path(self, out: Path) -> Path:
        return out / self.task / f"{self.id}.predictions.csv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "cell": self.cell,
            "method": self.method,
            "split_key": self.split_key,
            "fit_seed": self.fit_seed,
        }


@dataclass
class Grid:
    path: Path
    data: dict[str, Any]

    def file(self, key: str) -> Path:
        return self.path.parent / self.data[key]

    @property
    def out(self) -> Path:
        return self.file("out")

    @property
    def coverage_path(self) -> Path:
        return self.out.parent / f"{self.out.name}.coverage.json"


def _import_methods_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"accbench_methods_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise GridError(f"cannot load methods module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_grid(path: str | os.PathLike[str]) -> Grid:
    p = Path(path)
    if not p.is_file():
        raise GridError(f"grid {p} not found")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise GridError(f"grid {p} must be a YAML mapping")
    problems = [f"{k}: missing" for k in REQUIRED if k not in data]
    problems += [f"{k}: unknown key" for k in sorted(set(data) - GRID_KEYS)]
    methods = data.get("methods")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        problems.append("methods: must be a non-empty list of distinct method names")
    seeds = data.get("fit_seeds", [0])
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
    ):
        problems.append("fit_seeds: must be a non-empty list of integers")
    threads = data.get("threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        problems.append("threads: must be a positive integer")
    if problems:
        raise GridError(f"grid {p} is not valid: " + "; ".join(problems))
    grid = Grid(path=p, data=dict(data))
    if "methods_module" in data:
        _import_methods_module(grid.file("methods_module"))
    unknown = [m for m in methods if m not in available_methods()]
    if unknown:
        raise GridError(f"unknown method(s) {unknown}; available: {list(available_methods())}")
    return grid


def open_fixture(path: Path, fixtures_dir: Path) -> Any:
    """A grid names a frozen fixture directory, or the spec it was frozen from."""
    if path.is_file() and path.suffix in (".yaml", ".yml"):
        return locate_fixture(path, out_dir=fixtures_dir)
    return load_fixture(path)


@dataclass
class Context:
    grid: Grid
    fixture: Any
    registry: Any
    construct: Any
    prices: Any
    code: CodeState
    tasks: list[str]
    split_keys: list[str]
    fit_seeds: list[int]
    threads: int

    def plan(self) -> list[RunKey]:
        keys = []
        for task in self.tasks:
            for cell in self.fixture.cells(task):
                for method in self.grid.data["methods"]:
                    for split_key in self.split_keys:
                        for seed in self.fit_seeds:
                            keys.append(RunKey(task, cell, method, split_key, seed))
        return keys


def prepare(grid: Grid, *, code: CodeState | None = None) -> Context:
    """Load and cross-check everything a grid names, before any run starts."""
    construct = load_construct(grid.file("construct"))
    fixture = open_fixture(
        grid.file("fixture"), grid.path.parent / grid.data.get("fixtures_dir", "fixtures")
    )
    fx_construct = fixture.manifest.data["construct"]
    if fx_construct["sha256"] != construct.sha256:
        raise GridError(
            f"fixture {fixture.hash[:12]} was frozen under construct version "
            f"{fx_construct['version']} ({fx_construct['sha256'][:12]}), but the construct is "
            f"now version {construct.version} ({construct.sha256[:12]}); freeze it again"
        )
    manifest = read_artifact(grid.file("channel_manifest"))
    if manifest.data["fixture_hash"] != fixture.hash:
        raise GridError("the channel manifest was checked against a different fixture")
    registry = load_registry(grid.file("registry"))
    if Path(manifest.inputs["registry"]["path"]).name != grid.file("registry").name:
        raise GridError("the channel manifest was written for a different registry")

    tasks = list(grid.data.get("tasks") or fixture.tasks)
    missing = [t for t in tasks if t not in fixture.tasks]
    if missing:
        raise GridError(f"task(s) {missing} are not in the fixture")
    split_keys = list(grid.data.get("split_keys") or [fixture.split_keys[0]])
    bad = [k for k in split_keys if k not in fixture.split_keys]
    if bad:
        raise GridError(f"split key(s) {bad} were not frozen; frozen keys: {fixture.split_keys}")
    prices = load_prices(grid.file("prices")) if "prices" in grid.data else None
    return Context(
        grid=grid,
        fixture=fixture,
        registry=registry,
        construct=construct,
        prices=prices,
        code=code or code_state([grid.path.parent]),
        tasks=tasks,
        split_keys=split_keys,
        fit_seeds=list(grid.data.get("fit_seeds", [0])),
        threads=int(grid.data.get("threads", 1)),
    )


def _inputs(ctx: Context) -> dict[str, Path]:
    inputs = {
        "fixture": ctx.fixture.path / "manifest.json",
        "channel_manifest": ctx.grid.file("channel_manifest"),
        "registry": ctx.grid.file("registry"),
    }
    if ctx.prices is not None:
        inputs["prices"] = ctx.grid.file("prices")
    return inputs


def _n_classes(ctx: Context, task: str) -> int:
    t = ctx.fixture.manifest.data["tasks"][task]
    if t["type"] == "binary":
        return 2
    labels = {int(k) for split in t["class_counts"].values() for k in split}
    return max(labels) + 1


def _threadpools() -> list[dict[str, Any]]:
    return [
        {"api": p.get("internal_api"), "threads": p.get("num_threads")} for p in threadpool_info()
    ]


def run_one(ctx: Context, key: RunKey, rows: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    out = ctx.grid.out
    assignments, features = rows
    in_cell = (assignments["cell"] == key.cell).to_numpy()
    a, f = assignments[in_cell].reset_index(drop=True), features[in_cell].reset_index(drop=True)
    split = a[f"split__{key.split_key}"]
    task_meta = ctx.fixture.manifest.data["tasks"][key.task]
    meter = Meter(ctx.prices)
    base = {
        **key.to_dict(),
        "task_type": task_meta["type"],
        "metric": ctx.construct.metric,
        "fixture_hash": ctx.fixture.hash,
        "threads": ctx.threads,
        "n": {s: int((split == s).sum()) for s in ("train", "validation", "test")},
    }

    def finish(payload: dict[str, Any], extra_inputs: Mapping[str, Path] = ()) -> dict[str, Any]:
        payload["cost"] = meter.to_dict()
        return write_artifact(
            key.record_path(out),
            payload,
            code=ctx.code,
            inputs={**_inputs(ctx), **dict(extra_inputs)},
        )

    method = make_method(key.method)
    reason = method.unavailable()
    if reason is None:
        kinds = method.channel_kinds
        channels = [
            n
            for n, ch in ctx.registry.for_task(key.task).items()
            if kinds is None or ch.kind in kinds
        ]
        if not channels:
            reason = f"no registered channel of kind {sorted(kinds or [])} for task {key.task!r}"
    if reason is not None:
        return finish({**base, "status": "skipped", "reason": reason})

    try:
        train = (split == "train").to_numpy()
        y = a["label"].to_numpy()
        n_classes = _n_classes(ctx, key.task)
        fz = Featurizer(ctx.registry, task=key.task, channels=channels, task_type=task_meta["type"])
        with threadpool_limits(limits=ctx.threads):
            pools = _threadpools()
            t0 = time.perf_counter()
            x_train = fz.fit_transform(f[train], split[train], y[train])
            method.fit(
                x_train,
                y[train],
                n_classes=n_classes,
                seed=key.fit_seed,
                threads=ctx.threads,
                meter=meter,
            )
            fit_seconds = time.perf_counter() - t0
            x_eval = fz.transform(f[~train])
            proba = np.asarray(method.predict_proba(x_eval, meter=meter), dtype=float)
        if proba.shape != (int((~train).sum()), n_classes):
            raise GridError(f"{key.method} returned probabilities of shape {proba.shape}")

        ev = a[~train].reset_index(drop=True)
        scores = proba[:, 1] if n_classes == 2 else proba
        pred = pd.DataFrame(
            {
                "id": ev["id"],
                "cluster": ev["cluster"],
                "split": split[~train].to_numpy(),
                "label": ev["label"],
            }
        )
        if n_classes == 2:
            pred["score"] = scores
        else:
            for k in range(n_classes):
                pred[f"p_{k}"] = proba[:, k]
        atomic_write_text(key.predictions_path(out), pred.to_csv(index=False, lineterminator="\n"))

        results = {}
        for s in ("validation", "test"):
            mask = (pred["split"] == s).to_numpy()
            if mask.sum() == 0 or pred.loc[mask, "label"].nunique() < 2:
                results[s] = {"score": None, "tie_share": None}
                continue
            sc = scores[mask]
            results[s] = {
                "score": metrics.score(ctx.construct.metric, pred.loc[mask, "label"], sc),
                "tie_share": metrics.tie_share(sc),
            }
        return finish(
            {
                **base,
                "status": "ok",
                "fit_seconds": round(fit_seconds, 4),
                "threadpools": pools,
                "channels": channels,
                "n_features": int(x_train.shape[1]),
                "results": results,
            },
            {"predictions": key.predictions_path(out)},
        )
    except Exception as exc:  # recorded, never swallowed
        return finish(
            {
                **base,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _existing(ctx: Context, key: RunKey) -> tuple[str, str | None]:
    path = key.record_path(ctx.grid.out)
    if not path.exists():
        return "missing", None
    try:
        art = read_artifact(path, expected_code=ctx.code)
    except ProvenanceError as exc:
        return "stale", str(exc)
    return art.data["status"], None


def run_grid(
    grid_path: str | os.PathLike[str],
    *,
    code: CodeState | None = None,
    on_stale: str = "refuse",
    progress: Callable[[RunKey, str], None] | None = None,
) -> dict[str, int]:
    """Run every planned run that has no current record. Returns counts by outcome."""
    if on_stale not in ("refuse", "rerun"):
        raise ValueError("on_stale must be 'refuse' or 'rerun'")
    ctx = prepare(load_grid(grid_path), code=code)
    plan = ctx.plan()
    state = {key: _existing(ctx, key) for key in plan}
    stale = [(k, why) for k, (s, why) in state.items() if s == "stale"]
    if stale and on_stale == "refuse":
        first = stale[0][1]
        raise GridError(
            f"{len(stale)} existing record(s) are stale (e.g. {first}); "
            "rerun them with on_stale='rerun' (CLI: --rerun-stale) rather than mixing code states"
        )
    counts = {"resumed": 0, "ok": 0, "skipped": 0, "error": 0}
    rows_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key in plan:
        status, _ = state[key]
        if status in ("ok", "skipped"):
            counts["resumed"] += 1
            continue
        if key.task not in rows_cache:
            rows_cache[key.task] = ctx.fixture.rows(key.task)
        record = run_one(ctx, key, rows_cache[key.task])
        counts[record["status"]] += 1
        if progress:
            progress(key, record["status"])
    return counts


def coverage(
    grid_path: str | os.PathLike[str], *, code: CodeState | None = None, write: bool = True
) -> dict[str, Any]:
    """Count every planned run by status and write ``<out>.coverage.json``."""
    grid = load_grid(grid_path)
    ctx = prepare(grid, code=code)
    plan = ctx.plan()
    runs, by_method, counts = [], {}, dict.fromkeys(STATUSES, 0)
    for key in plan:
        status, why = _existing(ctx, key)
        reason = why
        if status in ("skipped", "error"):
            reason = read_artifact(key.record_path(grid.out)).data.get("reason")
        counts[status] += 1
        m = by_method.setdefault(key.method, dict.fromkeys(STATUSES, 0))
        m[status] += 1
        runs.append({**key.to_dict(), "status": status, "reason": reason})
    assert sum(counts.values()) == len(plan)
    payload = {
        "planned": len(plan),
        "counts": counts,
        "complete": counts["error"] == counts["missing"] == counts["stale"] == 0,
        "by_method": by_method,
        "runs": runs,
    }
    if write:
        return write_artifact(
            grid.coverage_path, payload, code=ctx.code, inputs={"grid": grid.path, "runs": grid.out}
        )
    return payload


def require_complete(cov: Mapping[str, Any]) -> None:
    """Refuse a grid in which any run is missing, failed or stale."""
    if cov["complete"]:
        return
    c = cov["counts"]
    bad = [r for r in cov["runs"] if r["status"] in ("error", "missing", "stale")]
    examples = ", ".join(f"{r['cell']}/{r['method']} ({r['status']})" for r in bad[:5])
    raise IncompleteGrid(
        f"{len(bad)} of {cov['planned']} planned runs are not complete "
        f"(error {c['error']}, missing {c['missing']}, stale {c['stale']}), e.g. {examples}; "
        "results over a partial grid would average each method over different cells"
    )
