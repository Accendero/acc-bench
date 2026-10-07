"""Metrics a construct statement may choose, and how each is computed.

``scores`` is a 1-D array of positive-class scores for a binary task, or an
``(n, k)`` array of class probabilities for a multiclass task. Multiclass ranking
metrics are macro-averaged one-vs-rest.

Ties: ``pr_auc`` is scikit-learn's average precision, which breaks tied scores as one
step. Section 13 of the paper found that convention worth 0.036 PR-AUC for a model
that emitted many tied scores, so :func:`tie_share` is recorded beside every score.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata
from sklearn import metrics as skm

HIGHER_IS_BETTER = {
    "pr_auc": True,
    "roc_auc": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "f1": True,
    "macro_f1": True,
    "brier": False,
    "log_loss": False,
}
METRICS = frozenset(HIGHER_IS_BETTER)


def _labels(scores: np.ndarray) -> np.ndarray:
    return (scores >= 0.5).astype(int) if scores.ndim == 1 else scores.argmax(axis=1)


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def _binary_check(y: np.ndarray) -> None:
    if y.min() == y.max():
        raise ValueError("only one class present in y_true")


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Binary average precision, equal to scikit-learn's (tied scores form one step).

    Written out because the bootstrap calls it thousands of times per cell.
    """
    _binary_check(y)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_of_tie]
    fp = (last_of_tie + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Binary ROC AUC by the rank-sum formula (ties count half), equal to scikit-learn's."""
    _binary_check(y)
    ranks = rankdata(s)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    if s.ndim == 1:
        return average_precision(y, s)
    return float(skm.average_precision_score(_one_hot(y, s.shape[1]), s, average="macro"))


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if s.ndim == 1:
        return roc_auc(y, s)
    return float(skm.roc_auc_score(y, s, multi_class="ovr", labels=np.arange(s.shape[1])))


def _brier(y: np.ndarray, s: np.ndarray) -> float:
    if s.ndim == 1:
        return float(np.mean((s - y) ** 2))
    return float(np.mean(np.sum((s - _one_hot(y, s.shape[1])) ** 2, axis=1)))


def _log_loss(y: np.ndarray, s: np.ndarray) -> float:
    labels = [0, 1] if s.ndim == 1 else list(range(s.shape[1]))
    return float(skm.log_loss(y, s, labels=labels))


_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pr_auc": _pr_auc,
    "roc_auc": _roc_auc,
    "accuracy": lambda y, s: float(skm.accuracy_score(y, _labels(s))),
    "balanced_accuracy": lambda y, s: float(skm.balanced_accuracy_score(y, _labels(s))),
    "f1": lambda y, s: float(skm.f1_score(y, _labels(s))),
    "macro_f1": lambda y, s: float(skm.f1_score(y, _labels(s), average="macro")),
    "brier": _brier,
    "log_loss": _log_loss,
}


def score(metric: str, y_true: np.ndarray, scores: np.ndarray) -> float:
    if metric not in _FUNCS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    return _FUNCS[metric](np.asarray(y_true).astype(int), np.asarray(scores, dtype=float))


def tie_share(scores: np.ndarray) -> float:
    """Share of rows whose positive-class score equals another row's."""
    s = np.asarray(scores, dtype=float)
    s = s if s.ndim == 1 else s.max(axis=1)
    if len(s) == 0:
        return 0.0
    _, counts = np.unique(s, return_counts=True)
    return float(counts[counts > 1].sum() / len(s))
