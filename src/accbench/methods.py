"""Methods the runner can fit, and the contract a new one follows.

A method declares which kinds of channel it reads (``channel_kinds``, or ``None`` for
all), says whether it can run here (:meth:`Method.unavailable` returns a reason or
``None``), and fits and predicts on the feature matrix the featurizer builds. A method
that calls a billable model records each call on the meter it is given.

Built in, all scikit-learn: ``majority``, ``logreg``, ``hist_gbm``, ``tfidf_logreg``.
Register others with :func:`register`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from accbench.costs import Meter


class Method:
    name: str = "method"
    channel_kinds: frozenset[str] | None = None

    def unavailable(self) -> str | None:
        """Return why this method cannot run here, or None if it can."""
        return None

    def fit(
        self,
        X: sp.csr_matrix,
        y: np.ndarray,
        *,
        n_classes: int,
        seed: int,
        threads: int,
        meter: Meter,
    ) -> Method:
        raise NotImplementedError

    def predict_proba(self, X: sp.csr_matrix, *, meter: Meter) -> np.ndarray:
        """Return an ``(n, n_classes)`` array of class probabilities."""
        raise NotImplementedError


class SklearnMethod(Method):
    dense = False

    def make(self, seed: int, threads: int) -> Any:
        raise NotImplementedError

    def fit(self, X, y, *, n_classes, seed, threads, meter):
        self.n_classes = n_classes
        self.model = self.make(seed, threads)
        self.model.fit(X.toarray() if self.dense else X, y)
        return self

    def predict_proba(self, X, *, meter):
        proba = self.model.predict_proba(X.toarray() if self.dense else X)
        # A class absent from the training rows gets probability 0, in its own column.
        out = np.zeros((proba.shape[0], self.n_classes))
        out[:, np.asarray(self.model.classes_, dtype=int)] = proba
        return out


class Majority(SklearnMethod):
    name = "majority"

    def make(self, seed, threads):
        return DummyClassifier(strategy="prior")


class LogReg(SklearnMethod):
    name = "logreg"
    channel_kinds = frozenset({"numeric", "categorical", "multihot"})

    def make(self, seed, threads):
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)


class HistGBM(SklearnMethod):
    name = "hist_gbm"
    channel_kinds = frozenset({"numeric", "categorical", "multihot"})
    dense = True

    def make(self, seed, threads):
        return HistGradientBoostingClassifier(random_state=seed)


class TfidfLogReg(SklearnMethod):
    name = "tfidf_logreg"
    channel_kinds = frozenset({"text"})

    def make(self, seed, threads):
        return LogisticRegression(C=4.0, max_iter=2000, random_state=seed)


_REGISTRY: dict[str, Callable[[], Method]] = {
    m.name: m for m in (Majority, LogReg, HistGBM, TfidfLogReg)
}


def register(factory: Callable[[], Method], name: str | None = None) -> None:
    """Make a method available to the runner under ``name`` (default: its ``name``)."""
    key = name or factory().name
    _REGISTRY[key] = factory


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def make_method(name: str) -> Method:
    if name not in _REGISTRY:
        raise KeyError(name)
    return _REGISTRY[name]()
