"""Bootstrap resampling clustered on the unit that repeats.

Rows that share a cluster (the same trial scored under several seeds, or several rows
of one trial) are drawn together. Resampling rows instead treats them as independent
and makes intervals too narrow: section 11 of the paper found a pooled bootstrap that
counted each trial once per seed made intervals about 2.17 times too narrow and
flipped a verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n_boot: int
    n_failed: int = 0

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "half_width": self.half_width,
            "n_boot": self.n_boot,
            "n_failed": self.n_failed,
        }


def cluster_index(clusters: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by cluster, in a fixed order."""
    clusters = np.asarray(clusters).astype(str)
    order = np.argsort(clusters, kind="stable")
    sorted_c = clusters[order]
    bounds = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1], True])
    return [order[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


def cluster_bootstrap_indices(
    clusters: np.ndarray, n_boot: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """``n_boot`` resamples of row indices, each drawing whole clusters with replacement."""
    clusters = np.asarray(clusters).astype(str)
    order = np.argsort(clusters, kind="stable")
    sorted_c = clusters[order]
    starts = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1]])
    lengths = np.diff(np.r_[starts, len(sorted_c)])
    k = len(starts)
    out = []
    for _ in range(n_boot):
        picks = rng.integers(0, k, k)
        lens = lengths[picks]
        offsets = np.cumsum(lens) - lens
        positions = np.repeat(starts[picks] - offsets, lens) + np.arange(lens.sum())
        out.append(order[positions])
    return out


def _interval(point: float, stats: np.ndarray, n_boot: int, level: float) -> Interval:
    ok = stats[~np.isnan(stats)]
    if len(ok) == 0:
        return Interval(point, float("nan"), float("nan"), n_boot, n_boot)
    alpha = (1 - level) / 2
    lo, hi = np.percentile(ok, [100 * alpha, 100 * (1 - alpha)])
    return Interval(point, float(lo), float(hi), n_boot, int(n_boot - len(ok)))


def _safe(metric: Metric, y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(metric(y, s))
    except ValueError:
        return float("nan")


def bootstrap_metric(
    metric: Metric,
    y: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Clustered percentile interval for one metric on one set of predictions."""
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(seed)
    stats = np.array(
        [_safe(metric, y[i], scores[i]) for i in cluster_bootstrap_indices(clusters, n_boot, rng)]
    )
    return _interval(_safe(metric, y, scores), stats, n_boot, level)


def paired_difference(
    metric: Metric,
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Clustered interval for metric(a) - metric(b) on the same rows, resampled together."""
    y, a, b = np.asarray(y), np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    stats = []
    for i in cluster_bootstrap_indices(clusters, n_boot, rng):
        stats.append(_safe(metric, y[i], a[i]) - _safe(metric, y[i], b[i]))
    point = _safe(metric, y, a) - _safe(metric, y, b)
    return _interval(point, np.array(stats), n_boot, level)


def difference_in_differences(
    metric: Metric,
    first: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Interval for [metric(a)-metric(b)] on ``first`` minus the same on ``second``.

    Each group is ``(y, scores_a, scores_b, clusters)`` and is resampled by its own
    clusters. This compares two drops with each other, which is the right question when
    a model's drop must be read against a reference's (section 11, T28b).
    """
    rng = np.random.default_rng(seed)

    def drop(group, idx=None):
        y, a, b, _ = (np.asarray(g) for g in group)
        if idx is not None:
            y, a, b = y[idx], a[idx], b[idx]
        return _safe(metric, y, a) - _safe(metric, y, b)

    boots_1 = cluster_bootstrap_indices(first[3], n_boot, rng)
    boots_2 = cluster_bootstrap_indices(second[3], n_boot, rng)
    stats = np.array(
        [drop(first, i) - drop(second, j) for i, j in zip(boots_1, boots_2, strict=True)]
    )
    return _interval(drop(first) - drop(second), stats, n_boot, level)
