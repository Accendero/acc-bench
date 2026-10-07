"""Unit 2, channels (R2 isolation): nothing reaches a model except through a registered path.

The channel registry (``channels.yaml``) assigns every column of every source table to
exactly one channel, or to ``ignore`` with a reason. There is no default branch: a
column the registry does not name raises :class:`UnregisteredColumn`, and a registered
column the table does not have raises too, because a misspelt name is how the
campaign's largest defect began (section 8). The label column may not sit in a channel.

Registry::

    channels:
      tabular:  {kind: numeric, columns: [age, enrollment], impute: median, scale: true}
      site:     {kind: categorical, columns: [city], encoding: one_hot, min_frequency: 5}
      summary:  {kind: text, columns: [summary], max_features: 20000, ngram_max: 2}
      codes:    {kind: multihot, columns: [icd_codes], separator: ";"}
    ignore:
      trial_id: the id
      died: the label source

Every channel may also carry ``tasks: [..]`` to apply to some tasks only.

The :class:`Featurizer` is fitted on training rows only. ``fit_transform`` refuses any
row whose split is not ``train``; target encoding is cross-fitted on those rows, so a
value seen once in training cannot point at its own label.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from accbench.errors import ChannelError, TrainOnlyViolation, UnregisteredColumn
from accbench.provenance import CodeState, code_state, write_artifact

_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

KINDS = {
    "numeric": {"impute": ("median", "mean", "zero"), "scale": bool, "missing_indicator": bool},
    "categorical": {"encoding": ("one_hot", "target"), "min_frequency": int},
    "text": {"max_features": int, "ngram_max": int, "min_df": int},
    "multihot": {"separator": str, "min_df": int},
}
REQUIRED_PARAMS = {"categorical": ("encoding",)}
DEFAULTS = {
    "numeric": {"impute": "median", "scale": True, "missing_indicator": False},
    "categorical": {"min_frequency": 1},
    "text": {"max_features": 20000, "ngram_max": 1, "min_df": 1},
    "multihot": {"separator": ";", "min_df": 1},
}


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str
    columns: tuple[str, ...]
    params: dict[str, Any]
    tasks: tuple[str, ...] | None = None

    def applies_to(self, task: str | None) -> bool:
        return task is None or self.tasks is None or task in self.tasks


def validate_registry(data: Any) -> list[str]:
    """Return every problem with a registry's own structure."""
    if not isinstance(data, Mapping):
        return ["the registry must be a YAML mapping"]
    problems = [f"{k}: unknown key" for k in sorted(set(data) - {"channels", "ignore"})]
    channels = data.get("channels")
    if not isinstance(channels, Mapping) or not channels:
        problems.append("channels: must map at least one channel name to its definition")
        channels = {}
    ignore = data.get("ignore") or {}
    if not isinstance(ignore, Mapping):
        problems.append("ignore: must map column names to the reason each is kept out")
        ignore = {}
    for col, reason in ignore.items():
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"ignore.{col}: give the reason the column is kept out")

    owner: dict[str, str] = {col: "ignore" for col in ignore}
    for name, ch in channels.items():
        where = f"channels.{name}"
        if not isinstance(ch, Mapping):
            problems.append(f"{where}: must be a mapping")
            continue
        kind = ch.get("kind")
        if kind not in KINDS:
            problems.append(f"{where}.kind: must be one of {sorted(KINDS)}")
            continue
        cols = ch.get("columns")
        if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
            problems.append(f"{where}.columns: must be a non-empty list of column names")
            cols = []
        for col in cols:
            if col in owner:
                problems.append(
                    f"column {col!r} is in both {owner[col]} and {name}; one channel per column"
                )
            else:
                owner[col] = name
        tasks = ch.get("tasks")
        if tasks is not None and not (
            isinstance(tasks, list) and tasks and all(isinstance(t, str) for t in tasks)
        ):
            problems.append(f"{where}.tasks: must be a non-empty list of task names")
        allowed = KINDS[kind]
        for key in sorted(set(ch) - {"kind", "columns", "tasks"} - set(allowed)):
            problems.append(f"{where}.{key}: not a setting of a {kind} channel")
        for key in REQUIRED_PARAMS.get(kind, ()):
            if key not in ch:
                problems.append(f"{where}.{key}: required for a {kind} channel (no default)")
        for key, rule in allowed.items():
            if key not in ch:
                continue
            value = ch[key]
            if isinstance(rule, tuple):
                if value not in rule:
                    problems.append(f"{where}.{key}: must be one of {list(rule)}")
            elif rule is bool:
                if not isinstance(value, bool):
                    problems.append(f"{where}.{key}: must be true or false")
            elif rule is int:
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    problems.append(f"{where}.{key}: must be a positive integer")
            elif not isinstance(value, rule):
                problems.append(f"{where}.{key}: must be a {rule.__name__}")
    return problems


@dataclass(frozen=True)
class Registry:
    path: Path | None
    channels: dict[str, Channel]
    ignore: dict[str, str]

    @classmethod
    def from_dict(cls, data: Any, path: Path | None = None) -> Registry:
        problems = validate_registry(data)
        if problems:
            raise ChannelError(f"channel registry {path or ''} is not valid".strip(), problems)
        channels = {}
        for name, ch in data["channels"].items():
            params = {**DEFAULTS[ch["kind"]]}
            params.update({k: v for k, v in ch.items() if k not in ("kind", "columns", "tasks")})
            channels[name] = Channel(
                name=name,
                kind=ch["kind"],
                columns=tuple(ch["columns"]),
                params=params,
                tasks=tuple(ch["tasks"]) if ch.get("tasks") else None,
            )
        return cls(path=path, channels=channels, ignore=dict(data.get("ignore") or {}))

    def for_task(self, task: str | None) -> dict[str, Channel]:
        return {n: c for n, c in self.channels.items() if c.applies_to(task)}

    def owner(self, column: str, task: str | None = None) -> str | None:
        if column in self.ignore:
            return "ignore"
        for name, ch in self.for_task(task).items():
            if column in ch.columns:
                return name
        return None

    def check_table(
        self,
        columns: Sequence[str],
        *,
        task: str | None = None,
        label_columns: Sequence[str] = (),
    ) -> None:
        """Refuse a table with an unregistered column, a missing registered column, or a
        label column inside a channel. Every problem is reported at once."""
        present = set(columns)
        problems: list[str] = []
        unregistered = [c for c in columns if self.owner(c, task) is None]
        for col in unregistered:
            problems.append(
                f"column {col!r} is not registered to any channel and not ignored; "
                "register it or ignore it with a reason"
            )
        for name, ch in self.for_task(task).items():
            for col in ch.columns:
                if col not in present:
                    problems.append(f"channel {name!r} registers {col!r}, which the table lacks")
        for col in label_columns:
            owner = self.owner(col, task)
            if owner not in (None, "ignore"):
                problems.append(f"label column {col!r} sits in channel {owner!r}; it would leak")
        if problems:
            where = f" for task {task!r}" if task else ""
            cls = UnregisteredColumn if unregistered else ChannelError
            raise cls(f"the table does not match the channel registry{where}", problems)


def load_registry(path: str | os.PathLike[str]) -> Registry:
    p = Path(path)
    if not p.is_file():
        raise ChannelError(f"channel registry {p} not found")
    return Registry.from_dict(yaml.safe_load(p.read_text(encoding="utf-8")), p)


# ---------------------------------------------------------------- featurizer


def _text(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    parts = [frame[c].fillna("").astype(str) for c in columns]
    out = parts[0]
    for part in parts[1:]:
        out = out + " " + part
    return out


def _target_encoder(task_type: str) -> TargetEncoder:
    # Cross-fitted with a fixed shuffle. sklearn 1.9 moved the shuffle into ``cv``.
    if _SKLEARN >= (1, 9):
        cv_cls = StratifiedKFold if task_type in ("binary", "multiclass") else KFold
        return TargetEncoder(target_type=task_type, cv=cv_cls(5, shuffle=True, random_state=0))
    return TargetEncoder(target_type=task_type, shuffle=True, random_state=0)


def _build(ch: Channel, task_type: str) -> Any:
    p = ch.params
    if ch.kind == "numeric":
        strategy = {"median": "median", "mean": "mean", "zero": "constant"}[p["impute"]]
        return [
            SimpleImputer(
                strategy=strategy,
                fill_value=0.0,
                add_indicator=p["missing_indicator"],
                keep_empty_features=True,
            ),
            StandardScaler() if p["scale"] else None,
        ]
    if ch.kind == "categorical":
        if p["encoding"] == "one_hot":
            return OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=p["min_frequency"],
                sparse_output=True,
            )
        return _target_encoder(task_type)
    if ch.kind == "text":
        return TfidfVectorizer(
            max_features=p["max_features"], ngram_range=(1, p["ngram_max"]), min_df=p["min_df"]
        )
    sep = p["separator"]
    return CountVectorizer(
        tokenizer=lambda s: [t for t in (x.strip() for x in s.split(sep)) if t],
        token_pattern=None,
        lowercase=False,
        binary=True,
        min_df=p["min_df"],
    )


class Featurizer:
    """Turns registered columns into a feature matrix, fitted on training rows only.

    ``channels`` picks which channels this featurizer uses (default: all that apply to
    the task); columns of other channels are left out on purpose, never by default.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        task: str | None = None,
        channels: Sequence[str] | None = None,
        task_type: str = "binary",
    ):
        available = registry.for_task(task)
        names = list(channels) if channels is not None else list(available)
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ChannelError(f"channels {unknown} are not in the registry for task {task!r}")
        if not names:
            raise ChannelError("a featurizer needs at least one channel")
        self.registry = registry
        self.task = task
        self.task_type = task_type
        self.channels = [available[n] for n in names]
        self._fitted: list[tuple[Channel, Any]] | None = None
        self.feature_names_: list[str] = []

    def _check(self, frame: pd.DataFrame) -> None:
        self.registry.check_table(list(frame.columns), task=self.task)

    def fit_transform(
        self, frame: pd.DataFrame, splits: pd.Series, y: Sequence[Any]
    ) -> sp.csr_matrix:
        """Fit on ``frame`` and return its features. Every row must be in the train split."""
        splits = pd.Series(splits)
        if len(splits) != len(frame):
            raise TrainOnlyViolation("splits must give one split per row")
        bad = splits[splits != "train"]
        if len(bad):
            counts = bad.value_counts().to_dict()
            raise TrainOnlyViolation(
                f"refusing to fit on {len(bad)} non-training row(s) {counts}; "
                "fit on the train split only"
            )
        self._check(frame)
        y_arr = np.asarray(y)
        blocks, fitted, names = [], [], []
        for ch in self.channels:
            est = _build(ch, self.task_type)
            if ch.kind == "numeric":
                imputer, scaler = est
                block = imputer.fit_transform(frame[list(ch.columns)].astype(float))
                if scaler is not None:
                    block = scaler.fit_transform(block)
                out_names = [f"{ch.name}:{n}" for n in imputer.get_feature_names_out(ch.columns)]
                block = sp.csr_matrix(block)
            elif ch.kind == "categorical":
                values = frame[list(ch.columns)].astype("string").fillna("<missing>").astype(object)
                if isinstance(est, TargetEncoder):
                    block = sp.csr_matrix(est.fit_transform(values, y_arr))  # cross-fitted
                else:
                    block = sp.csr_matrix(est.fit_transform(values))
                out_names = [f"{ch.name}:{n}" for n in est.get_feature_names_out(ch.columns)]
            else:
                block = sp.csr_matrix(est.fit_transform(_text(frame, ch.columns)))
                out_names = [f"{ch.name}:{n}" for n in est.get_feature_names_out()]
            blocks.append(block)
            fitted.append((ch, est))
            names.extend(out_names)
        self._fitted = fitted
        self.feature_names_ = names
        return sp.hstack(blocks, format="csr")

    def transform(self, frame: pd.DataFrame) -> sp.csr_matrix:
        if self._fitted is None:
            raise ChannelError("transform called before fit_transform")
        self._check(frame)
        blocks = []
        for ch, est in self._fitted:
            if ch.kind == "numeric":
                imputer, scaler = est
                block = imputer.transform(frame[list(ch.columns)].astype(float))
                if scaler is not None:
                    block = scaler.transform(block)
                blocks.append(sp.csr_matrix(block))
            elif ch.kind == "categorical":
                values = frame[list(ch.columns)].astype("string").fillna("<missing>").astype(object)
                blocks.append(sp.csr_matrix(est.transform(values)))
            else:
                blocks.append(sp.csr_matrix(est.transform(_text(frame, ch.columns))))
        return sp.hstack(blocks, format="csr")


# ---------------------------------------------------------------- the manifest


def label_columns(fixture: Any, task: str) -> list[str]:
    rule = fixture.manifest.data["tasks"][task]["label_rule"]
    return [rule["column"]] if rule.get("kind") == "column" else []


def write_manifest(
    registry_path: str | os.PathLike[str],
    fixture: Any,
    out: str | os.PathLike[str],
    *,
    code: CodeState | None = None,
) -> dict[str, Any]:
    """Check every task's source table against the registry and write the channel manifest.

    The manifest records, per task, the channel each column takes into a model, and is
    stamped over the registry and the fixture so a later change to either makes it stale.
    """
    registry = load_registry(registry_path)
    tasks = {}
    for task in fixture.tasks:
        source = pd.read_csv(fixture.source_path(task), nrows=0)
        cols = list(source.columns)
        registry.check_table(cols, task=task, label_columns=label_columns(fixture, task))
        tasks[task] = {
            "columns": {c: registry.owner(c, task) for c in cols},
            "channels": {
                n: {"kind": ch.kind, "columns": list(ch.columns), "params": ch.params}
                for n, ch in registry.for_task(task).items()
            },
        }
    payload = {"fixture_hash": fixture.hash, "ignore": registry.ignore, "tasks": tasks}
    return write_artifact(
        out,
        payload,
        code=code or code_state([Path(registry_path).parent]),
        inputs={"registry": registry_path, "fixture": fixture.path / "manifest.json"},
    )
