"""Cost metering against a pinned price table.

The price table (``prices.yaml``) pins the price of every billable model, with the date
it was read and where::

    currency: USD
    models:
      some-model:
        input_per_1k: 0.003
        output_per_1k: 0.015
        read_on: 2026-10-01
        source_url: https://example.com/pricing

A call to a model the table does not price fails closed with :class:`UnpricedModel`,
so nothing is run at an unknown cost. Every run record carries the table's sha256.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from accbench.errors import AccbenchError
from accbench.provenance import file_digest


class PriceError(AccbenchError):
    """The price table is invalid."""


class UnpricedModel(PriceError):
    """A call was made to a model the price table does not price."""


@dataclass(frozen=True)
class PriceTable:
    path: Path | None
    sha256: str | None
    currency: str
    models: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Any, path: Path | None = None) -> PriceTable:
        problems = []
        if not isinstance(data, Mapping):
            raise PriceError("the price table must be a YAML mapping")
        models = data.get("models")
        if not isinstance(models, Mapping) or not models:
            problems.append("models: must price at least one model")
            models = {}
        for name, m in models.items():
            if not isinstance(m, Mapping):
                problems.append(f"models.{name}: must be a mapping")
                continue
            for key in ("input_per_1k", "output_per_1k"):
                v = m.get(key)
                if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                    problems.append(f"models.{name}.{key}: must be a price of at least 0")
            if not isinstance(m.get("read_on"), (dt.date, str)):
                problems.append(f"models.{name}.read_on: give the date the price was read")
            if not isinstance(m.get("source_url"), str) or not m.get("source_url"):
                problems.append(f"models.{name}.source_url: give where the price was read")
        if problems:
            raise PriceError("price table is not valid: " + "; ".join(problems))
        return cls(
            path=path,
            sha256=file_digest(path) if path else None,
            currency=str(data.get("currency", "USD")),
            models={k: dict(v) for k, v in models.items()},
        )

    def price(self, model: str) -> dict[str, Any]:
        if model not in self.models:
            raise UnpricedModel(
                f"model {model!r} is not in the price table"
                f"{f' {self.path}' if self.path else ''}; price it before calling it"
            )
        return self.models[model]


def load_prices(path: str | os.PathLike[str]) -> PriceTable:
    p = Path(path)
    if not p.is_file():
        raise PriceError(f"price table {p} not found")
    return PriceTable.from_dict(yaml.safe_load(p.read_text(encoding="utf-8")), p)


@dataclass
class Meter:
    """Counts calls, tokens and cost per model. A meter with no table refuses every call."""

    prices: PriceTable | None = None
    calls: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, model: str, *, input_tokens: int, output_tokens: int) -> float:
        if self.prices is None:
            raise UnpricedModel(f"no price table was given, so model {model!r} cannot be called")
        p = self.prices.price(model)
        cost = input_tokens / 1000 * p["input_per_1k"] + output_tokens / 1000 * p["output_per_1k"]
        row = self.calls.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        )
        row["calls"] += 1
        row["input_tokens"] += input_tokens
        row["output_tokens"] += output_tokens
        row["cost"] += cost
        return cost

    @property
    def total(self) -> float:
        return float(sum(r["cost"] for r in self.calls.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.prices.currency if self.prices else None,
            "price_table_sha256": self.prices.sha256 if self.prices else None,
            "total": self.total,
            "by_model": {k: dict(v) for k, v in sorted(self.calls.items())},
        }
