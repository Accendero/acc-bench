"""Unit 5, rules (R5 sound conclusions): decision rules as code, each tested on a known answer.

A decision rule turns measurements into a verdict for one pre-registered question.
Rules are functions registered with :func:`rule`, and every rule carries a **null
input**: a generator of synthetic data that contains no effect. Before any verdict is
written, the null-input suite runs each rule on many null draws and refuses a rule that
claims an effect where there is none (section 11). The suite is stamped with the code
state and each rule's source hash, so it reruns whenever a rule changes.

A rule sees the data through a :class:`RuleContext`:

- ``ctx.goodness(y, scores)``: the construct metric, signed so that larger is better;
- ``ctx.floor(cell)``: the cell's noise floor from the resolution table;
- ``ctx.test_rows(cell, method, arm=None)``: labels, scores and clusters on the test set;
- ``ctx.select(cell, candidates)``: picks a method **on validation scores only**;
- ``ctx.oracle_select(cell, candidates)``: picks on test scores, and marks the verdict as
  an oracle, so it can never pass for a confirmatory result.

Outcomes are ``effect`` (with a direction), ``no_effect`` (the interval sits inside the
floor) or ``inconclusive``.

Built-in rules: ``difference_clears_floor``, ``k_of_n_cells``, ``difference_of_drops``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from accbench import metrics
from accbench.errors import AccbenchError
from accbench.provenance import (
    CodeState,
    code_state,
    read_artifact,
    write_artifact,
)
from accbench.resampling import difference_in_differences, paired_difference
from accbench.resolution import read_resolution, score_columns
from accbench.runner import RunKey, load_grid, prepare

OUTCOMES = ("effect", "no_effect", "inconclusive")


class RuleError(AccbenchError):
    """A rule or question is invalid."""


class NullInputFailure(RuleError):
    """A rule claimed an effect on data that contains none."""


class SelectionOnTest(RuleError):
    """A selection tried to use test data outside the oracle path."""


# ---------------------------------------------------------------- the context a rule sees


class RuleContext(Protocol):
    def goodness(self, y: np.ndarray, scores: np.ndarray) -> float: ...
    def floor(self, cell: str) -> float: ...
    def test_rows(
        self, cell: str, method: str, arm: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def select(self, cell: str, candidates: Sequence[str]) -> str: ...
    def oracle_select(self, cell: str, candidates: Sequence[str]) -> str: ...


@dataclass
class Verdict:
    outcome: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise RuleError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")


@dataclass
class _Tracking:
    selections: list[dict[str, Any]] = field(default_factory=list)
    oracle: bool = False


def _pick(scores: Mapping[str, float]) -> str:
    return max(sorted(scores), key=lambda m: scores[m])


class SyntheticContext:
    """A context over in-memory data, used for null inputs and tests.

    ``rows[(cell, method)]`` is ``(y, scores, clusters, arms)``; ``validation[(cell,
    method)]`` is a validation goodness; ``floors[cell]`` is the floor.
    """

    def __init__(self, rows, floors, validation=None, metric="roc_auc"):
        self.rows = rows
        self.floors = floors
        self.validation = validation or {}
        self.metric = metric
        self.tracking = _Tracking()

    def goodness(self, y, scores):
        value = metrics.score(self.metric, y, scores)
        return value if metrics.HIGHER_IS_BETTER[self.metric] else -value

    def floor(self, cell):
        return self.floors[cell]

    def test_rows(self, cell, method, arm=None):
        y, s, c, arms = self.rows[(cell, method)]
        if arm is None:
            return y, s, c
        mask = np.asarray(arms) == arm
        return y[mask], s[mask], c[mask]

    def select(self, cell, candidates):
        chosen = _pick({m: self.validation[(cell, m)] for m in candidates})
        self.tracking.selections.append(
            {"cell": cell, "candidates": list(candidates), "chosen": chosen, "on": "validation"}
        )
        return chosen

    def oracle_select(self, cell, candidates):
        chosen = _pick({m: self.goodness(*self.test_rows(cell, m)[:2]) for m in candidates})
        self.tracking.selections.append(
            {"cell": cell, "candidates": list(candidates), "chosen": chosen, "on": "test (oracle)"}
        )
        self.tracking.oracle = True
        return chosen


class BenchContext:
    """A context over a finished grid and its resolution table."""

    def __init__(self, grid_ctx: Any, resolution: Any):
        self.ctx = grid_ctx
        self.resolution = resolution
        self.metric = grid_ctx.construct.metric
        self._floors = {r["cell"]: r["floor"] for r in resolution.data["cells"]}
        self._key = (grid_ctx.split_keys[0], grid_ctx.fit_seeds[0])
        self._arms: dict[str, pd.Series] = {}
        self.tracking = _Tracking()

    def goodness(self, y, scores):
        value = metrics.score(self.metric, y, scores)
        return value if metrics.HIGHER_IS_BETTER[self.metric] else -value

    def floor(self, cell):
        if cell not in self._floors:
            raise RuleError(f"cell {cell!r} is not in the resolution table")
        return self._floors[cell]

    def _record(self, cell, method, split_key, seed):
        key = RunKey(cell.split("/")[0], cell, method, split_key, seed)
        art = read_artifact(key.record_path(self.ctx.grid.out), expected_code=self.ctx.code)
        if art.data["status"] != "ok":
            raise RuleError(f"{cell}/{method} has no completed run ({art.data['status']})")
        return key, art

    def _arm(self, task: str) -> pd.Series:
        if task not in self._arms:
            a = self.ctx.fixture.assignments(task)
            self._arms[task] = a.set_index("id")["arm"]
        return self._arms[task]

    def test_rows(self, cell, method, arm=None):
        key, _ = self._record(cell, method, *self._key)
        pred = pd.read_csv(
            key.predictions_path(self.ctx.grid.out), dtype={"id": str, "cluster": str}
        )
        test = pred[pred["split"] == "test"]
        if arm is not None:
            arms = self._arm(key.task).reindex(test["id"]).to_numpy()
            test = test[arms == arm]
        return test["label"].to_numpy(), score_columns(test), test["cluster"].to_numpy()

    def _validation(self, cell, method):
        values = []
        for split_key in self.ctx.split_keys:
            for seed in self.ctx.fit_seeds:
                _, art = self._record(cell, method, split_key, seed)
                v = art.data["results"]["validation"]["score"]
                if v is not None:
                    values.append(v if metrics.HIGHER_IS_BETTER[self.metric] else -v)
        if not values:
            raise RuleError(f"{cell}/{method} has no validation score to select on")
        return float(np.mean(values))

    def select(self, cell, candidates):
        chosen = _pick({m: self._validation(cell, m) for m in candidates})
        self.tracking.selections.append(
            {"cell": cell, "candidates": list(candidates), "chosen": chosen, "on": "validation"}
        )
        return chosen

    def oracle_select(self, cell, candidates):
        chosen = _pick({m: self.goodness(*self.test_rows(cell, m)[:2]) for m in candidates})
        self.tracking.selections.append(
            {"cell": cell, "candidates": list(candidates), "chosen": chosen, "on": "test (oracle)"}
        )
        self.tracking.oracle = True
        return chosen


# ---------------------------------------------------------------- registering rules


NullInput = Callable[[int], tuple[RuleContext, dict[str, Any]]]


@dataclass(frozen=True)
class Rule:
    name: str
    func: Callable[..., Verdict]
    null_input: NullInput
    exact_null: NullInput | None = None

    @property
    def source_sha256(self) -> str:
        parts = []
        for f in (self.func, self.null_input, self.exact_null):
            if f is None:
                continue
            try:
                parts.append(inspect.getsource(f))
            except (OSError, TypeError):
                parts.append(repr(f))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()


_RULES: dict[str, Rule] = {}


def rule(name: str, *, null_input: NullInput, exact_null: NullInput | None = None):
    """Register a decision rule with its null input (and, optionally, an exact null)."""

    def wrap(func: Callable[..., Verdict]) -> Callable[..., Verdict]:
        _RULES[name] = Rule(name, func, null_input, exact_null)
        return func

    return wrap


def get_rule(name: str) -> Rule:
    if name not in _RULES:
        raise RuleError(f"unknown rule {name!r}; registered: {sorted(_RULES)}")
    return _RULES[name]


def registered_rules() -> tuple[str, ...]:
    return tuple(sorted(_RULES))


def resolve_method(ctx: RuleContext, cell: str, spec: Any) -> str:
    """A method is named directly, or chosen on validation with ``{select: [..]}``."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping) and set(spec) == {"select"}:
        return ctx.select(cell, list(spec["select"]))
    if isinstance(spec, Mapping) and set(spec) == {"oracle_select"}:
        return ctx.oracle_select(cell, list(spec["oracle_select"]))
    raise RuleError(
        f"a method is a name, {{select: [..]}} or {{oracle_select: [..]}}; got {spec!r}"
    )


# ---------------------------------------------------------------- built-in rules


def _clusters(n):
    return np.array([f"c{i}" for i in range(n)])


def _null_pair(seed: int, *, n: int = 240, floor: float = 0.04):
    """Two methods with the same skill, scored with independent noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = y * 0.8
    a = signal + rng.normal(0, 1, n)
    b = signal + rng.normal(0, 1, n)
    rows = {("c", "a"): (y, a, _clusters(n), None), ("c", "b"): (y, b, _clusters(n), None)}
    ctx = SyntheticContext(rows, {"c": floor}, {("c", "a"): 0.7, ("c", "b"): 0.7})
    return ctx, {"cell": "c", "a": "a", "b": "b", "n_boot": 200, "seed": seed}


def _exact_pair(seed: int):
    ctx, params = _null_pair(seed)
    y, a, c, _ = ctx.rows[("c", "a")]
    ctx.rows[("c", "b")] = (y, a.copy(), c, None)
    return ctx, params


@rule("difference_clears_floor", null_input=_null_pair, exact_null=_exact_pair)
def difference_clears_floor(
    ctx: RuleContext, *, cell: str, a: Any, b: Any, n_boot: int = 1000, seed: int = 0
) -> Verdict:
    """Method a against method b in one cell: an effect needs an interval that excludes zero
    and a point difference larger than the cell's floor."""
    ma, mb = resolve_method(ctx, cell, a), resolve_method(ctx, cell, b)
    y, sa, clusters = ctx.test_rows(cell, ma)
    y_b, sb, _ = ctx.test_rows(cell, mb)
    if len(y) != len(y_b) or not np.array_equal(y, y_b):
        raise RuleError(f"{ma} and {mb} were not scored on the same test rows in {cell}")
    iv = paired_difference(ctx.goodness, y, sa, sb, clusters, n_boot=n_boot, seed=seed)
    floor = ctx.floor(cell)
    details = {"a": ma, "b": mb, "difference": iv.to_dict(), "floor": floor}
    if iv.excludes_zero() and abs(iv.point) > floor:
        better = ma if iv.point > 0 else mb
        return Verdict(
            "effect",
            f"{better} is better by {abs(iv.point):.4f} (floor {floor:.4f})",
            {**details, "better": better},
        )
    if -floor < iv.lo and iv.hi < floor:
        return Verdict("no_effect", f"the difference lies inside the floor {floor:.4f}", details)
    return Verdict(
        "inconclusive",
        f"difference {iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}], floor {floor:.4f}",
        details,
    )


def _null_cells(seed: int, *, n_cells: int = 16):
    rng = np.random.default_rng(seed)
    rows, floors, val = {}, {}, {}
    for i in range(n_cells):
        n = 150
        y = rng.integers(0, 2, n)
        for m in ("a", "b"):
            rows[(f"c{i}", m)] = (y, y * 0.8 + rng.normal(0, 1, n), _clusters(n), None)
            val[(f"c{i}", m)] = 0.7
        floors[f"c{i}"] = 0.05
    ctx = SyntheticContext(rows, floors, val)
    cells = [f"c{i}" for i in range(n_cells)]
    return ctx, {"cells": cells, "a": "a", "b": "b", "k": 5, "n_boot": 100, "seed": seed}


@rule("k_of_n_cells", null_input=_null_cells)
def k_of_n_cells(
    ctx: RuleContext,
    *,
    cells: Sequence[str],
    a: Any,
    b: Any,
    k: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> Verdict:
    """a beats b in at least k of the cells, a cell counting only when its paired interval
    excludes zero and the difference exceeds the cell's floor.

    Counting point differences above the floor alone failed this rule's null-input test:
    a difference between two methods is noisier than one method's floor, so on data with
    no effect it claimed one in 4 of 20 draws.
    """
    clears, per_cell = [], {}
    for cell in cells:
        ma, mb = resolve_method(ctx, cell, a), resolve_method(ctx, cell, b)
        y, sa, clusters = ctx.test_rows(cell, ma)
        _, sb, _ = ctx.test_rows(cell, mb)
        iv = paired_difference(ctx.goodness, y, sa, sb, clusters, n_boot=n_boot, seed=seed)
        floor = ctx.floor(cell)
        per_cell[cell] = {"a": ma, "b": mb, "difference": iv.to_dict(), "floor": floor}
        if iv.lo > 0 and iv.point > floor:
            clears.append(cell)
    details = {"k": k, "n": len(cells), "clears": clears, "per_cell": per_cell}
    if len(clears) >= k:
        return Verdict(
            "effect",
            f"a beats b beyond the floor in {len(clears)} of {len(cells)} "
            f"cells (rule: {k} or more)",
            details,
        )
    return Verdict(
        "no_effect",
        f"{len(clears)} of {len(cells)} cells clear the floor (rule: {k} or more)",
        details,
    )


def _null_drops(seed: int, *, n: int = 300):
    """Both models are worse on the unseen arm by the same amount: no memorization."""
    rng = np.random.default_rng(seed)
    rows = {}
    for arm, strength in (("seen", 1.0), ("unseen", 0.6)):
        y = rng.integers(0, 2, n)
        rows[(arm, "model")] = (y, y * strength + rng.normal(0, 1, n))
        rows[(arm, "reference")] = (y, y * strength + rng.normal(0, 1, n))
    data = {}
    for m in ("model", "reference"):
        y = np.concatenate([rows[("seen", m)][0], rows[("unseen", m)][0]])
        s = np.concatenate([rows[("seen", m)][1], rows[("unseen", m)][1]])
        arms = np.array(["seen"] * n + ["unseen"] * n)
        data[("c", m)] = (y, s, _clusters(2 * n), arms)
    # Both methods must share labels; reuse the model's labels for the reference.
    y_m = data[("c", "model")][0]
    _, s_r, c_r, arms_r = data[("c", "reference")]
    data[("c", "reference")] = (y_m, s_r, c_r, arms_r)
    ctx = SyntheticContext(data, {"c": 0.05})
    params = {
        "cell": "c",
        "model": "model",
        "reference": "reference",
        "seen_arm": "seen",
        "unseen_arm": "unseen",
        "n_boot": 200,
        "seed": seed,
    }
    return ctx, params


@rule("difference_of_drops", null_input=_null_drops)
def difference_of_drops(
    ctx: RuleContext,
    *,
    cell: str,
    model: str,
    reference: str,
    seen_arm: str,
    unseen_arm: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Verdict:
    """Does the model's advantage over the reference shrink on data it could not have seen?
    The two drops are compared with each other; neither is read alone (section 11, T28b)."""
    groups = []
    for arm in (seen_arm, unseen_arm):
        y, sm, cl = ctx.test_rows(cell, model, arm)
        y_r, sr, _ = ctx.test_rows(cell, reference, arm)
        if not np.array_equal(y, y_r):
            raise RuleError(f"{model} and {reference} were not scored on the same rows of {arm}")
        groups.append((y, sm, sr, cl))
    iv = difference_in_differences(ctx.goodness, groups[0], groups[1], n_boot=n_boot, seed=seed)
    details = {"advantage_seen_minus_unseen": iv.to_dict()}
    if iv.excludes_zero():
        word = "larger" if iv.point > 0 else "smaller"
        return Verdict(
            "effect",
            f"the model's advantage is {word} on the seen arm by "
            f"{iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}]",
            details,
        )
    return Verdict(
        "inconclusive",
        f"difference of drops {iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}] includes zero",
        details,
    )


# ---------------------------------------------------------------- the null-input suite


MAX_EFFECT_RATE = 0.15
MIN_GATE_DRAWS = 20


def run_null_suite(
    names: Sequence[str] | None = None, *, draws: int = 20, max_effect_rate: float = MAX_EFFECT_RATE
) -> dict[str, Any]:
    """Run each rule on ``draws`` null inputs. A rule passes if it never claims an effect on
    its exact null and claims one on at most ``floor(draws * max_effect_rate)`` sampled null
    draws (3 of 20 by default, against a nominal 1 in 20)."""
    if draws < 1:
        raise RuleError("the null-input suite needs at least one draw")
    max_effects = int(draws * max_effect_rate)
    results = {}
    for name in names or registered_rules():
        r = get_rule(name)
        effects = []
        for d in range(draws):
            ctx, params = r.null_input(d)
            if r.func(ctx, **params).outcome == "effect":
                effects.append(d)
        exact_effect = None
        if r.exact_null is not None:
            ctx, params = r.exact_null(0)
            exact_effect = r.func(ctx, **params).outcome == "effect"
        passed = len(effects) <= max_effects and not exact_effect
        results[name] = {
            "source_sha256": r.source_sha256,
            "draws": draws,
            "effects_on_null": len(effects),
            "max_effects": max_effects,
            "exact_null_claimed_effect": exact_effect,
            "passed": passed,
        }
    return results


def ensure_null_suite(
    names: Sequence[str], path: Path, *, code: CodeState, draws: int = 20
) -> dict[str, Any]:
    """Reuse a current suite record for these rules, or run the suite and write one.
    Raises :class:`NullInputFailure` if any rule fails.

    This is the gate before any verdict, so it refuses fewer than ``MIN_GATE_DRAWS``
    draws: with a handful, a sound rule fails by chance and an unsound one can pass.
    """
    if draws < MIN_GATE_DRAWS:
        raise RuleError(f"the gate needs at least {MIN_GATE_DRAWS} null draws, got {draws}")
    current: dict[str, Any] = {}
    if path.exists():
        try:
            art = read_artifact(path, expected_code=code)
            current = art.data["rules"]
        except AccbenchError:
            current = {}
    needed = [
        n
        for n in names
        if n not in current
        or current[n]["source_sha256"] != get_rule(n).source_sha256
        or current[n]["draws"] < draws
    ]
    if needed:
        current = {**current, **run_null_suite(needed, draws=draws)}
        write_artifact(path, {"rules": current}, code=code)
    failed = [n for n in names if not current[n]["passed"]]
    if failed:
        detail = "; ".join(
            f"{n}: effect on {current[n]['effects_on_null']} of {current[n]['draws']} null draws"
            + (", and on its exact null" if current[n]["exact_null_claimed_effect"] else "")
            for n in failed
        )
        raise NullInputFailure(f"rule(s) fail the null-input test, no verdict written: {detail}")
    return current


# ---------------------------------------------------------------- questions and verdicts


def _load_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"accbench_rules_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuleError(f"cannot load rules module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_questions(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise RuleError(f"questions file {p} not found")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    problems = []
    if not isinstance(data, Mapping):
        raise RuleError(f"{p} must be a YAML mapping")
    for key in ("grid", "resolution", "questions"):
        if key not in data:
            problems.append(f"{key}: missing")
    for key in sorted(set(data) - {"grid", "resolution", "questions", "rules_module"}):
        problems.append(f"{key}: unknown key")
    if "rules_module" in data:
        _load_module(p.parent / data["rules_module"])
    ids = []
    for i, q in enumerate(data.get("questions") or []):
        where = f"questions[{i}]"
        if not isinstance(q, Mapping):
            problems.append(f"{where}: must be a mapping")
            continue
        for key in ("id", "question", "rule", "params"):
            if key not in q:
                problems.append(f"{where}.{key}: missing")
        if q.get("rule") is not None and q["rule"] not in _RULES:
            problems.append(f"{where}.rule: unknown rule {q['rule']!r}")
        ids.append(q.get("id"))
    if not data.get("questions"):
        problems.append("questions: list at least one question")
    dups = sorted({i for i in ids if ids.count(i) > 1 and i is not None})
    if dups:
        problems.append(f"questions: duplicate id(s) {dups}")
    if problems:
        raise RuleError(f"questions file {p} is not valid: " + "; ".join(problems))
    return dict(data)


def decide(
    questions_path: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    code: CodeState | None = None,
    draws: int = 20,
) -> dict[str, Any]:
    """Run the null-input suite for the rules in use, then answer every question."""
    qpath = Path(questions_path)
    q = load_questions(qpath)
    grid = load_grid(qpath.parent / q["grid"])
    gctx = prepare(grid, code=code or code_state([qpath.parent]))
    code = gctx.code
    res_path = qpath.parent / q["resolution"]
    resolution = read_resolution(res_path, expected_code=code)
    if resolution.data["fixture_hash"] != gctx.fixture.hash:
        raise RuleError("the resolution table was computed on a different fixture")

    names = sorted({question["rule"] for question in q["questions"]})
    suite_path = qpath.parent / "null_suite.json"
    suite = ensure_null_suite(names, suite_path, code=code, draws=draws)

    verdicts = []
    for question in q["questions"]:
        r = get_rule(question["rule"])
        ctx = BenchContext(gctx, resolution)
        v = r.func(ctx, **question["params"])
        verdicts.append(
            {
                "id": question["id"],
                "question": question["question"],
                "rule": r.name,
                "rule_sha256": r.source_sha256,
                "params": question["params"],
                "outcome": v.outcome,
                "summary": v.summary,
                "details": v.details,
                "selections": ctx.tracking.selections,
                "oracle": ctx.tracking.oracle,
            }
        )
    payload = {
        "metric": gctx.construct.metric,
        "null_suite": {n: suite[n] for n in names},
        "verdicts": verdicts,
    }
    target = Path(out) if out else qpath.parent / "verdicts.json"
    return write_artifact(
        target,
        payload,
        code=code,
        inputs={
            "questions": qpath,
            "resolution": res_path,
            "runs": grid.out,
            "null_suite": suite_path,
        },
    )
