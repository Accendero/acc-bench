"""Unit 1, fixtures (R1 sameness): every method meets the same frozen splits, label rule and timing.

A fixture spec (``fixtures.yaml``) names, for each task in the construct statement, the
source table, the one rule that makes its label, and how rows group into cells. It also
declares the split and, optionally, cutoff dates and the arms defined against them.
:func:`freeze` applies the spec and writes ``fixtures/<hash>/``:

- ``<task>.csv``: one row per kept source row with its id, cluster, cell, label, its
  split under every split key, and its arm;
- ``manifest.json``: what was frozen and when, the counts behind it, every excluded
  row with its reason, and a provenance stamp over the spec and the source tables.

The directory name is a hash of the content, so a fixture cannot change in place: a
different freeze lands in a different directory. :func:`load_fixture` refuses a
fixture whose files or source tables have changed since it was frozen.

Splits are keyed: a row's split comes from a hash of the split key and its cluster id,
so it does not depend on row order and adding rows never moves existing ones. The test
set has its own key (or comes from a column in the source), so it stays fixed while the
train/validation split is redrawn under each key in ``split.keys``. That makes a redraw
a separate, named setting, and lets the resolution unit attribute variance to it.

Spec::

    construct: construct.yaml
    id_column: trial_id
    cluster_column: trial_id        # optional; the unit that repeats (default id_column)
    tasks:
      mortality:
        source: data/mortality.csv  # relative to this file
        type: binary                # or multiclass
        label_rule: {column: died}  # or "module:function", taking the table, returning labels
        cell_by: phase              # optional
    split:
      test: {fraction: 0.2, key: test-v1}    # or {column: split, value: test}
      validation: {fraction: 0.2}            # drawn from the non-test rows
      keys: [split-0, split-1, split-2]      # the first is the primary split
    cutoffs: {model: 2025-01-01}             # optional
    arms:                                    # optional
      A: {cutoff: model, before: [results_posted_on]}
      B: {cutoff: model, before: [registered_on], after: [results_posted_on]}
      C: {cutoff: model, after: [registered_on]}
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from accbench.construct import Construct, require_construct
from accbench.errors import FixtureError, StaleInput
from accbench.provenance import (
    Artifact,
    CodeState,
    atomic_write_text,
    code_state,
    file_digest,
    read_artifact,
    write_artifact,
)

MANIFEST = "manifest.json"
TASK_TYPES = ("binary", "multiclass")
SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------- the spec


def _date(value: Any) -> dt.date | None:
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


def _fraction(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 1


def validate_spec(spec: Any, construct: Construct | None = None) -> list[str]:
    """Return every problem with a fixture spec; an empty list means it passes."""
    if not isinstance(spec, Mapping):
        return ["the spec must be a YAML mapping"]
    problems: list[str] = []
    allowed = {"construct", "id_column", "cluster_column", "tasks", "split", "cutoffs", "arms"}
    for key in ("construct", "id_column", "tasks", "split"):
        if key not in spec:
            problems.append(f"{key}: missing")
    for key in sorted(set(spec) - allowed):
        problems.append(f"{key}: unknown key")

    tasks = spec.get("tasks")
    if "tasks" in spec:
        if not isinstance(tasks, Mapping) or not tasks:
            problems.append("tasks: must map at least one task name to its definition")
            tasks = {}
        for name, t in tasks.items():
            where = f"tasks.{name}"
            if construct is not None and name not in construct.tasks:
                problems.append(
                    f"{where}: not in the construct statement, which names "
                    f"{list(construct.tasks)}; freeze only what the construct names"
                )
            if not isinstance(t, Mapping):
                problems.append(f"{where}: must be a mapping")
                continue
            for key in ("source", "type", "label_rule"):
                if key not in t:
                    problems.append(f"{where}.{key}: missing")
            allowed_task = {"source", "type", "label_rule", "label_columns", "cell_by"}
            for key in sorted(set(t) - allowed_task):
                problems.append(f"{where}.{key}: unknown key")
            if "type" in t and t["type"] not in TASK_TYPES:
                problems.append(f"{where}.type: must be one of {list(TASK_TYPES)}")
            rule = t.get("label_rule")
            if "label_rule" in t:
                ok_fn = isinstance(rule, str) and rule.count(":") == 1
                ok_col = (
                    isinstance(rule, Mapping)
                    and isinstance(rule.get("column"), str)
                    and set(rule) <= {"column", "map"}
                    and (rule.get("map") is None or isinstance(rule.get("map"), Mapping))
                )
                if not (ok_fn or ok_col):
                    problems.append(
                        f"{where}.label_rule: must be 'module:function' or "
                        "{column: name, map: {...}} (one rule, no fallbacks)"
                    )
                cols = t.get("label_columns")
                if ok_fn and not (
                    isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols)
                ):
                    problems.append(
                        f"{where}.label_columns: a function label rule must name the columns "
                        "it reads, so no model can read them"
                    )
                if ok_col and "label_columns" in t:
                    problems.append(
                        f"{where}.label_columns: only a function label rule takes label_columns"
                    )

    split = spec.get("split")
    if "split" in spec:
        if not isinstance(split, Mapping):
            problems.append("split: must be a mapping")
        else:
            for key in sorted(set(split) - {"test", "validation", "keys"}):
                problems.append(f"split.{key}: unknown key")
            test = split.get("test")
            if not isinstance(test, Mapping) or not (
                (
                    set(test) == {"fraction", "key"}
                    and _fraction(test.get("fraction"))
                    and isinstance(test.get("key"), str)
                    and test.get("key")
                )
                or (set(test) == {"column", "value"} and isinstance(test.get("column"), str))
            ):
                problems.append(
                    "split.test: must be {fraction: 0<f<1, key: name} or {column: name, value: v}"
                )
            val = split.get("validation")
            if (
                not isinstance(val, Mapping)
                or set(val) != {"fraction"}
                or not _fraction(val.get("fraction"))
            ):
                problems.append("split.validation: must be {fraction: 0<f<1}")
            keys = split.get("keys")
            if (
                not isinstance(keys, list)
                or not keys
                or not all(isinstance(k, str) and k for k in keys)
                or len(set(keys)) != len(keys)
            ):
                problems.append("split.keys: must be a non-empty list of distinct names")

    cutoffs = spec.get("cutoffs") or {}
    if not isinstance(cutoffs, Mapping):
        problems.append("cutoffs: must map names to dates")
        cutoffs = {}
    for name, value in cutoffs.items():
        if _date(value) is None:
            problems.append(f"cutoffs.{name}: must be a date (YYYY-MM-DD)")

    arms = spec.get("arms") or {}
    if not isinstance(arms, Mapping):
        problems.append("arms: must map arm names to definitions")
        arms = {}
    for name, arm in arms.items():
        where = f"arms.{name}"
        if not isinstance(arm, Mapping):
            problems.append(f"{where}: must be a mapping")
            continue
        for key in sorted(set(arm) - {"cutoff", "before", "after"}):
            problems.append(f"{where}.{key}: unknown key")
        if arm.get("cutoff") not in cutoffs:
            problems.append(f"{where}.cutoff: must name one of the cutoffs {list(cutoffs)}")
        cols = [*(arm.get("before") or []), *(arm.get("after") or [])]
        if not cols or not all(isinstance(c, str) for c in cols):
            problems.append(f"{where}: needs date columns under 'before' and/or 'after'")
    return problems


# ---------------------------------------------------------------- splits, labels, arms


def unit_hash(key: str, value: object) -> float:
    """A uniform number in [0, 1) fixed by the key and the value, on every machine."""
    digest = hashlib.sha256(f"{key}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _uniform(key: str, values: pd.Series) -> np.ndarray:
    return np.fromiter((unit_hash(key, v) for v in values), dtype=float, count=len(values))


def assign_splits(
    clusters: pd.Series,
    *,
    keys: list[str],
    test: Mapping[str, Any],
    validation_fraction: float,
    test_column: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Assign train/validation/test under each split key. The test set is the same for all keys."""
    if "column" in test:
        if test_column is None:
            raise FixtureError(f"split.test names column {test['column']!r}, which is missing")
        is_test = (test_column.astype(str) == str(test["value"])).to_numpy()
    else:
        is_test = _uniform(f"test/{test['key']}", clusters) < float(test["fraction"])
    out = {}
    for key in keys:
        is_val = _uniform(f"validation/{key}", clusters) < validation_fraction
        split = np.where(is_test, "test", np.where(is_val, "validation", "train"))
        out[key] = pd.Series(split, index=clusters.index, name=f"split__{key}")
    return out


def _import_rule(ref: str, search: Path) -> Callable[[pd.DataFrame], Any]:
    module_name, func_name = ref.split(":")
    local = search / (module_name.replace(".", "/") + ".py")
    if local.is_file():
        # Load the spec's own file under a private name, so two projects with a
        # labels.py each never share a cached module.
        private = "accbench_rule_" + hashlib.sha256(str(local.resolve()).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(private, local)
        if spec is None or spec.loader is None:
            raise FixtureError(f"label rule {ref!r}: cannot load {local}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise FixtureError(f"label rule {ref!r}: cannot import {module_name}") from exc
    func = getattr(module, func_name, None)
    if not callable(func):
        raise FixtureError(f"label rule {ref!r}: {func_name} is not a function in {module_name}")
    return func


def _same_labels(first: Any, second: Any) -> bool:
    a = pd.to_numeric(pd.Series(np.asarray(first)), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(np.asarray(second)), errors="coerce").to_numpy(dtype=float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def apply_label_rule(
    rule: Any,
    table: pd.DataFrame,
    *,
    search: Path,
    label_columns: list[str] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply the task's one label rule. Returns the labels and a description of the rule."""
    if isinstance(rule, Mapping):
        col = rule["column"]
        if col not in table.columns:
            raise FixtureError(f"label rule column {col!r} is not in the source table")
        labels = table[col]
        if rule.get("map") is not None:
            mapping = dict(rule["map"])
            labels = labels.map(lambda v: mapping.get(v, np.nan) if pd.notna(v) else np.nan)
        desc = {"kind": "column", "column": col}
        if rule.get("map") is not None:
            desc["map"] = json.dumps(rule["map"], sort_keys=True, default=str)
        return pd.Series(labels, index=table.index), desc

    func = _import_rule(rule, search)
    declared = list(label_columns or [])
    missing = [c for c in declared if c not in table.columns]
    if missing:
        raise FixtureError(f"label_columns names {missing}, which the source table lacks")
    labels = func(table.copy())
    # The function must give the same labels from the declared columns alone; if it does
    # not, it reads a column that label_columns does not name, and a model could read it.
    try:
        restricted = func(table[declared].copy())
    except (KeyError, IndexError) as exc:
        raise FixtureError(
            f"label rule {rule!r} reads a column that label_columns does not name ({exc})"
        ) from exc
    if not _same_labels(labels, restricted):
        raise FixtureError(
            f"label rule {rule!r} gives different labels from the columns in label_columns "
            "alone; it reads a column that label_columns does not name"
        )
    labels = pd.Series(labels, index=table.index) if not isinstance(labels, pd.Series) else labels
    if not labels.index.equals(table.index):
        raise FixtureError(f"label rule {rule!r} must return one label per row, in table order")
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ""
    desc = {
        "kind": "function",
        "ref": rule,
        "columns": declared,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    return labels, desc


def _check_labels(task: str, task_type: str, labels: pd.Series) -> pd.Series:
    kept = labels.dropna()
    try:
        as_int = kept.astype(float)
    except (TypeError, ValueError) as exc:
        raise FixtureError(f"{task}: labels must be integers; got {kept.unique()[:5]}") from exc
    if not np.all(np.equal(np.mod(as_int, 1), 0)):
        raise FixtureError(f"{task}: labels must be integers")
    as_int = as_int.astype(int)
    values = set(as_int.unique())
    if task_type == "binary" and not values <= {0, 1}:
        raise FixtureError(f"{task}: a binary task needs labels 0 and 1, got {sorted(values)}")
    if len(values) < 2:
        raise FixtureError(f"{task}: the label rule produced one class only ({sorted(values)})")
    return as_int


def assign_arms(
    table: pd.DataFrame, arms: Mapping[str, Any], cutoffs: Mapping[str, Any]
) -> pd.Series:
    """Assign each row to the one arm whose date conditions it meets, or to none."""
    arm = pd.Series([None] * len(table), index=table.index, dtype=object)
    for name, spec in arms.items():
        cutoff = pd.Timestamp(_date(cutoffs[spec["cutoff"]]))
        match = pd.Series(True, index=table.index)
        for side in ("before", "after"):
            for col in spec.get(side) or []:
                if col not in table.columns:
                    raise FixtureError(f"arm {name!r} needs date column {col!r}, which is missing")
                dates = pd.to_datetime(table[col], errors="coerce")
                cond = dates < cutoff if side == "before" else dates >= cutoff
                match &= cond.fillna(False) & dates.notna()
        clash = match & arm.notna()
        if clash.any():
            other = arm[clash].iloc[0]
            raise FixtureError(
                f"arms {other!r} and {name!r} overlap on {int(clash.sum())} row(s); "
                "arms must be disjoint"
            )
        arm[match] = name
    return arm


# ---------------------------------------------------------------- freezing


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _csv_text(frame: pd.DataFrame) -> str:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


@dataclass(frozen=True)
class Fixture:
    """A frozen fixture, checked on load."""

    path: Path
    manifest: Artifact

    @property
    def hash(self) -> str:
        return self.manifest.data["fixture_hash"]

    @property
    def frozen_at(self) -> str:
        return self.manifest.data["frozen_at"]

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.manifest.data["tasks"])

    @property
    def split_keys(self) -> tuple[str, ...]:
        return tuple(self.manifest.data["split"]["keys"])

    def cells(self, task: str) -> tuple[str, ...]:
        return tuple(self.manifest.data["tasks"][task]["cells"])

    def assignments(self, task: str) -> pd.DataFrame:
        return pd.read_csv(
            self.path / f"{task}.csv",
            dtype={"id": str, "cluster": str, "cell": str, "arm": str},
            keep_default_na=False,
            na_values={"arm": [""]},
        )

    def source_path(self, task: str) -> Path:
        stored = Path(self.manifest.inputs[f"source:{task}"]["path"])
        return stored if stored.is_absolute() else self.path / stored

    def rows(self, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The fixture's assignments and the matching source rows, aligned row for row.

        The source table was checked against its frozen hash when the fixture loaded.
        """
        id_col = self.manifest.data["id_column"]
        assignments = self.assignments(task)
        source = pd.read_csv(self.source_path(task), dtype={id_col: str}).set_index(id_col)
        features = source.loc[assignments["id"]].reset_index()
        return assignments, features

    def cell_table(self) -> str:
        """One line per cell: rows in train, validation and test under the primary key."""
        key = self.split_keys[0]
        lines = [f"  {'cell':30} {'train':>7} {'validation':>10} {'test':>7}   (split key {key})"]
        for t in self.manifest.data["tasks"].values():
            for cell, by_key in t["split_counts"].items():
                c = by_key[key]
                lines.append(f"  {cell:30} {c['train']:7d} {c['validation']:10d} {c['test']:7d}")
        return "\n".join(lines)

    def summary(self) -> str:
        parts = []
        for task, t in self.manifest.data["tasks"].items():
            parts.append(f"{task}: {t['rows_kept']} rows, {len(t['cells'])} cell(s)")
        return (
            f"fixture {self.hash[:12]} frozen {self.frozen_at}; "
            f"split keys {list(self.split_keys)}; " + "; ".join(parts)
        )


def freeze(
    spec_path: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str] = "fixtures",
    code: CodeState | None = None,
) -> Fixture:
    """Freeze the fixture a spec describes. Refuses without a valid construct statement."""
    return _freeze(spec_path, out_dir=out_dir, code=code, write=True)


def spec_cells(spec_path: str | os.PathLike[str]) -> set[str]:
    """The cell names a fixture spec gives, read from its source tables without a freeze."""
    spec_file = Path(spec_path)
    spec = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    cells: set[str] = set()
    for task, t in (spec.get("tasks") or {}).items():
        cell_by = t.get("cell_by")
        if cell_by is None:
            cells.add(task)
            continue
        values = pd.read_csv(spec_file.parent / t["source"], usecols=[cell_by])[cell_by]
        cells |= {f"{task}/{v}" for v in values.dropna().astype(str).unique()}
    return cells


def locate_fixture(
    spec_path: str | os.PathLike[str], *, out_dir: str | os.PathLike[str] = "fixtures"
) -> Fixture:
    """Load the frozen fixture that ``spec_path`` freezes to today, without writing anything.

    Refuses if it has not been frozen: a grid can name its spec instead of a hash, and
    still never runs on a fixture that was frozen as a side effect of running.
    """
    target = _freeze(spec_path, out_dir=out_dir, code=None, write=False)
    if not (target / MANIFEST).is_file():
        raise FixtureError(
            f"{spec_path} has not been frozen with its current spec, construct and data "
            f"(expected {target}); run acc-bench fixtures freeze first"
        )
    return load_fixture(target)


def _freeze(
    spec_path: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str],
    code: CodeState | None,
    write: bool,
) -> Any:
    spec_file = Path(spec_path)
    if not spec_file.is_file():
        raise FixtureError(f"fixture spec {spec_file} not found")
    base = spec_file.parent
    spec = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping) or "construct" not in spec:
        raise FixtureError(f"{spec_file}: the spec must name its construct statement")

    construct = require_construct(base / spec["construct"])
    problems = validate_spec(spec, construct)
    if problems:
        raise FixtureError(
            f"fixture spec {spec_file} is not valid:\n" + "\n".join(f"  - {p}" for p in problems)
        )

    id_col = spec["id_column"]
    cluster_col = spec.get("cluster_column", id_col)
    split = spec["split"]
    keys = list(split["keys"])
    cutoffs = spec.get("cutoffs") or {}
    arms = spec.get("arms") or {}

    files: dict[str, str] = {}
    sources: dict[str, Path] = {}
    task_meta: dict[str, Any] = {}
    for task, t in spec["tasks"].items():
        source = base / t["source"]
        if not source.is_file():
            raise FixtureError(f"{task}: source table {source} not found")
        sources[f"source:{task}"] = source
        table = pd.read_csv(source, dtype={id_col: str})
        for col in (id_col, cluster_col):
            if col not in table.columns:
                raise FixtureError(f"{task}: column {col!r} is not in {source.name}")
        if table[id_col].isna().any():
            raise FixtureError(f"{task}: {int(table[id_col].isna().sum())} row(s) have no id")
        dups = table[id_col][table[id_col].duplicated()]
        if len(dups):
            raise FixtureError(
                f"{task}: id {dups.iloc[0]!r} appears more than once; one row per id "
                "(use cluster_column for units that repeat)"
            )

        raw_labels, rule_desc = apply_label_rule(
            t["label_rule"], table, search=base, label_columns=t.get("label_columns")
        )
        excluded: dict[str, int] = {}
        missing = raw_labels.isna()
        if missing.any():
            excluded["label rule returned no label"] = int(missing.sum())
        kept = table.loc[~missing]
        labels = _check_labels(task, t["type"], raw_labels.loc[~missing])

        cell_by = t.get("cell_by")
        if cell_by is not None:
            if cell_by not in kept.columns:
                raise FixtureError(f"{task}: cell_by column {cell_by!r} is missing")
            no_cell = kept[cell_by].isna()
            if no_cell.any():
                excluded[f"no value for {cell_by}"] = int(no_cell.sum())
                kept, labels = kept.loc[~no_cell], labels.loc[~no_cell]
            cells = task + "/" + kept[cell_by].astype(str)
        else:
            cells = pd.Series(task, index=kept.index)

        clusters = kept[cluster_col].astype(str)
        test_col = None
        if "column" in split["test"]:
            test_col = kept.get(split["test"]["column"])
        splits = assign_splits(
            clusters,
            keys=keys,
            test=split["test"],
            validation_fraction=float(split["validation"]["fraction"]),
            test_column=test_col,
        )
        arm = assign_arms(kept, arms, cutoffs) if arms else pd.Series(None, index=kept.index)

        frame = pd.DataFrame(
            {
                "id": kept[id_col].astype(str),
                "cluster": clusters,
                "cell": cells,
                "label": labels,
                **{f"split__{k}": v for k, v in splits.items()},
                "arm": arm,
            }
        ).sort_values("id", kind="stable")
        text = _csv_text(frame)
        files[f"{task}.csv"] = text

        primary = f"split__{keys[0]}"
        counts = {
            cell: {key: {s: int((g[f"split__{key}"] == s).sum()) for s in SPLITS} for key in keys}
            for cell, g in frame.groupby("cell", sort=True)
        }
        for cell, g in frame.groupby("cell"):
            for s in SPLITS:
                part = g[g[primary] == s]
                if s != "validation" and part["label"].nunique() < 2:
                    raise FixtureError(
                        f"{cell}: the {s} split under key {keys[0]!r} has one class only"
                    )
        classes = {
            s: {
                str(k): int(v)
                for k, v in frame[frame[primary] == s]["label"].value_counts().sort_index().items()
            }
            for s in SPLITS
        }
        task_meta[task] = {
            "source": {"path": t["source"], "sha256": file_digest(source)},
            "type": t["type"],
            "label_rule": rule_desc,
            "cell_by": cell_by,
            "rows_in": int(len(table)),
            "rows_kept": int(len(frame)),
            "excluded": excluded,
            "cells": sorted(counts),
            "split_counts": counts,
            "class_counts": classes,
            "arm_counts": (
                {
                    str(k): int(v)
                    for k, v in frame["arm"].fillna("(none)").value_counts().sort_index().items()
                }
                if arms
                else {}
            ),
        }

    content = {
        "spec": spec,
        "construct_sha256": construct.sha256,
        "sources": {name: file_digest(path) for name, path in sorted(sources.items())},
        "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
    }
    fixture_hash = hashlib.sha256(_canonical(content).encode()).hexdigest()
    target = Path(out_dir) / fixture_hash[:16]
    if not write:
        return target

    if (target / MANIFEST).exists():
        existing = load_fixture(target)
        if existing.hash != fixture_hash:
            raise FixtureError(f"{target} holds a different fixture; refusing to overwrite it")
        return existing

    for name, text in files.items():
        atomic_write_text(target / name, text)
    payload = {
        "fixture_hash": fixture_hash,
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "construct": {
            "path": spec["construct"],
            "sha256": construct.sha256,
            "version": construct.version,
        },
        "id_column": id_col,
        "cluster_column": cluster_col,
        "split": {
            "keys": keys,
            "primary": keys[0],
            "test": dict(split["test"]),
            "validation": dict(split["validation"]),
        },
        "cutoffs": {k: str(_date(v)) for k, v in cutoffs.items()},
        "arms": {k: dict(v) for k, v in arms.items()},
        "tasks": task_meta,
        "files": content["files"],
    }
    write_artifact(
        target / MANIFEST,
        payload,
        code=code or code_state([base]),
        inputs={"spec": spec_file, **sources},
    )
    return load_fixture(target)


def load_fixture(
    path: str | os.PathLike[str],
    *,
    expected_code: CodeState | None = None,
    allow_code_drift: bool = False,
) -> Fixture:
    """Load a frozen fixture. Refuses one whose files, spec or source tables have changed."""
    p = Path(path)
    if not (p / MANIFEST).is_file():
        raise FixtureError(f"{p} is not a frozen fixture (no {MANIFEST})")
    art = read_artifact(
        p / MANIFEST, expected_code=expected_code, allow_code_drift=allow_code_drift
    )
    changed = []
    for name, sha in art.data["files"].items():
        f = p / name
        if not f.is_file():
            changed.append(f"{name} is missing")
        elif hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            changed.append(f"{name} changed")
    if changed:
        raise StaleInput(
            f"fixture {p} was modified after it was frozen ({'; '.join(changed)}); "
            "freeze a new one instead"
        )
    return Fixture(path=p, manifest=art)
