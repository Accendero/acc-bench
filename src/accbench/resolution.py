"""Unit 4, resolution (R4): the smallest difference each cell can resolve, by source.

For every cell, three sources of noise are measured separately:

- **split**: the spread of a method's test score across split keys, at a fixed fit seed
  (needs two or more split keys in the grid);
- **fit**: the spread across fit seeds, at a fixed split key (needs two or more seeds);
- **test**: the half-width of a 95 percent bootstrap interval on the test set,
  resampled by cluster (the unit that repeats).

Each term is a 95 percent half-width (1.96 standard deviations for the first two),
computed per method and then summarized across methods by the median. The cell's
noise floor is the largest term, and the table names the source that sets it. A term
the grid cannot measure is recorded as ``None``, never as zero.

:func:`resolve` refuses to build a table from a grid that is incomplete, from records
written under different code states, or from records whose code state differs from
the code running now. It writes ``resolution.json``, stamped over the run directory, so
a later re-run of any record makes the table stale and every reader refuses it.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from accbench import metrics
from accbench.errors import AccbenchError
from accbench.provenance import (
    Artifact,
    CodeState,
    read_artifact,
    require_same_code,
    write_artifact,
)
from accbench.resampling import bootstrap_metric
from accbench.runner import RunKey, coverage, load_grid, prepare, require_complete

Z95 = 1.96
TERMS = ("split", "fit", "test")


class ResolutionError(AccbenchError):
    """A resolution table cannot be built or read."""


def score_columns(pred: pd.DataFrame) -> np.ndarray:
    if "score" in pred.columns:
        return pred["score"].to_numpy(dtype=float)
    cols = sorted((c for c in pred.columns if c.startswith("p_")), key=lambda c: int(c[2:]))
    return pred[cols].to_numpy(dtype=float)


def _spread(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return None
    return float(Z95 * np.std(vals, ddof=1))


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _median(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def resolve(
    grid_path: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    methods: Sequence[str] | None = None,
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
    threshold: float = 0.05,
    code: CodeState | None = None,
) -> dict[str, Any]:
    """Compute the noise floor of every cell and write the resolution table."""
    grid = load_grid(grid_path)
    ctx = prepare(grid, code=code)
    cov = coverage(grid_path, code=ctx.code)
    require_complete(cov)

    chosen = list(methods) if methods else [m for m in grid.data["methods"] if m != "majority"]
    unknown = [m for m in chosen if m not in grid.data["methods"]]
    if unknown:
        raise ResolutionError(f"method(s) {unknown} are not in the grid")
    metric_name = ctx.construct.metric

    def metric_fn(y, s):
        return metrics.score(metric_name, y, s)

    records: list[Artifact] = []
    by_cell: dict[str, dict[str, list[Artifact]]] = defaultdict(lambda: defaultdict(list))
    skipped: dict[str, list[str]] = defaultdict(list)
    for run in cov["runs"]:
        if run["method"] not in chosen:
            continue
        key = RunKey(run["task"], run["cell"], run["method"], run["split_key"], run["fit_seed"])
        if run["status"] == "skipped":
            if run["method"] not in skipped[run["cell"]]:
                skipped[run["cell"]].append(run["method"])
            continue
        art = read_artifact(key.record_path(grid.out), expected_code=ctx.code)
        records.append(art)
        by_cell[run["cell"]][run["method"]].append(art)
    if not records:
        raise ResolutionError("no completed runs for the chosen methods; nothing to resolve")
    require_same_code(records)

    primary_key = ctx.split_keys[0]
    primary_seed = ctx.fit_seeds[0]
    rows = []
    for cell in sorted(by_cell):
        per_method = {}
        n_test = n_clusters = None
        for method, arts in sorted(by_cell[cell].items()):
            scores = {
                (a.data["split_key"], a.data["fit_seed"]): a.data["results"]["test"]["score"]
                for a in arts
            }
            split_terms = [
                _spread([scores.get((k, s)) for k in ctx.split_keys]) for s in ctx.fit_seeds
            ]
            fit_terms = [
                _spread([scores.get((k, s)) for s in ctx.fit_seeds]) for k in ctx.split_keys
            ]
            test_terms = []
            for a in arts:
                pred_path = a.path.with_name(a.path.stem + ".predictions.csv")
                pred = pd.read_csv(pred_path, dtype={"id": str, "cluster": str})
                test = pred[pred["split"] == "test"]
                iv = bootstrap_metric(
                    metric_fn,
                    test["label"].to_numpy(),
                    score_columns(test),
                    test["cluster"].to_numpy(),
                    n_boot=n_boot,
                    seed=bootstrap_seed,
                )
                test_terms.append(None if np.isnan(iv.half_width) else iv.half_width)
                if (a.data["split_key"], a.data["fit_seed"]) == (primary_key, primary_seed):
                    n_test = len(test)
                    n_clusters = int(test["cluster"].nunique())
            per_method[method] = {
                "split": _mean(split_terms),
                "fit": _mean(fit_terms),
                "test": _mean(test_terms),
                "mean_test_score": _mean(list(scores.values())),
                "runs": len(arts),
            }
        terms = {t: _median([m[t] for m in per_method.values()]) for t in TERMS}
        measured = {t: v for t, v in terms.items() if v is not None}
        if not measured:
            raise ResolutionError(f"{cell}: no noise term could be measured")
        source = max(measured, key=measured.get)
        floor = measured[source]
        rows.append(
            {
                "cell": cell,
                "task": cell.split("/")[0],
                "n_test": n_test,
                "n_test_clusters": n_clusters,
                "methods": sorted(per_method),
                "skipped_methods": sorted(skipped.get(cell, [])),
                "terms": terms,
                "floor": floor,
                "source": source,
                "resolves_threshold": floor <= threshold,
                "per_method": per_method,
            }
        )

    payload = {
        "metric": metric_name,
        "threshold": threshold,
        "terms_measured": {
            "split": len(ctx.split_keys) > 1,
            "fit": len(ctx.fit_seeds) > 1,
            "test": True,
        },
        "summary_across_methods": "median",
        "bootstrap": {
            "n_boot": n_boot,
            "seed": bootstrap_seed,
            "level": 0.95,
            "cluster": ctx.fixture.manifest.data["cluster_column"],
        },
        "split_keys": ctx.split_keys,
        "fit_seeds": ctx.fit_seeds,
        "fixture_hash": ctx.fixture.hash,
        "cells": rows,
        "underpowered": [r["cell"] for r in rows if not r["resolves_threshold"]],
    }
    target = Path(out) if out else grid.path.parent / "resolution.json"
    return write_artifact(
        target,
        payload,
        code=ctx.code,
        inputs={"grid": grid.path, "runs": grid.out},
    )


def read_resolution(
    path: str | os.PathLike[str], *, expected_code: CodeState | None = None
) -> Artifact:
    """Read a resolution table, refusing it if any run behind it changed or the code moved."""
    return read_artifact(path, expected_code=expected_code)


def floors(table: Artifact | Mapping[str, Any]) -> dict[str, float]:
    data = table.data if isinstance(table, Artifact) else table
    return {row["cell"]: row["floor"] for row in data["cells"]}
