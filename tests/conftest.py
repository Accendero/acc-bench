"""Shared synthetic data: a small trials table, a construct statement and a fixture spec."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

CONSTRUCT = {
    "version": 1,
    "stated_on": "2026-10-07",
    "owner": "A. Person",
    "question": "Which method should rank trials for mortality review?",
    "decision": "Which trials a reviewer reads first.",
    "target": {"name": "mortality", "definition": "Any death reported in the trial's results."},
    "metric": {"name": "pr_auc", "reason": "Only the top of the ranked list is acted on."},
    "tasks": [
        {"name": "mortality", "reason": "The decision concerns mortality."},
        {"name": "dropout", "reason": "Second task, for multi-task tests."},
    ],
    "reported_numbers": [{"name": "pr_auc_by_cell", "decision": "Which method to deploy."}],
}

SPEC = {
    "construct": "construct.yaml",
    "id_column": "trial_id",
    "tasks": {
        "mortality": {
            "source": "data/mortality.csv",
            "type": "binary",
            "label_rule": {"column": "died"},
            "cell_by": "phase",
        }
    },
    "split": {
        "test": {"fraction": 0.2, "key": "test-v1"},
        "validation": {"fraction": 0.2},
        "keys": ["split-0", "split-1", "split-2"],
    },
}


def make_trials(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 12, n).round(1)
    enrollment = rng.integers(10, 2000, n)
    phase = rng.choice(["Phase1", "Phase2", "Phase3"], n)
    logit = -1.0 + 0.04 * (age - 55) + rng.normal(0, 1, n)
    died = (logit > 0).astype(int)
    registered = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2900, n), unit="D")
    posted = registered + pd.to_timedelta(rng.integers(200, 1500, n), unit="D")
    return pd.DataFrame(
        {
            "trial_id": [f"NCT{i:08d}" for i in range(n)],
            "phase": phase,
            "age": age,
            "enrollment": enrollment,
            "summary": [f"trial of drug {i % 37} in adults" for i in range(n)],
            "died": died,
            "registered_on": registered.strftime("%Y-%m-%d"),
            "results_posted_on": posted.strftime("%Y-%m-%d"),
        }
    )


class Project:
    """A throwaway benchmark project laid out on disk."""

    def __init__(self, root):
        self.root = root
        self.construct = copy.deepcopy(CONSTRUCT)
        self.spec = copy.deepcopy(SPEC)
        self.table = make_trials()

    def write(self):
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "construct.yaml").write_text(yaml.safe_dump(self.construct, sort_keys=False))
        (self.root / "fixtures.yaml").write_text(yaml.safe_dump(self.spec, sort_keys=False))
        self.table.to_csv(self.root / "data" / "mortality.csv", index=False)
        return self

    @property
    def spec_path(self):
        return self.root / "fixtures.yaml"

    @property
    def out(self):
        return self.root / "fixtures"


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / "proj")


REGISTRY = {
    "channels": {
        "tabular": {"kind": "numeric", "columns": ["age", "enrollment"]},
        "phase": {"kind": "categorical", "columns": ["phase"], "encoding": "one_hot"},
        "summary": {"kind": "text", "columns": ["summary"]},
    },
    "ignore": {
        "trial_id": "the id",
        "died": "the label source",
        "registered_on": "used for arms only",
        "results_posted_on": "used for arms only",
    },
}

METHODS_MODULE = '''
from accbench.methods import Method, SklearnMethod, register
from sklearn.linear_model import LogisticRegression


class PricedStub(SklearnMethod):
    """Stands in for a billable model: every predict call is metered."""
    name = "priced_stub"
    channel_kinds = frozenset({"text"})

    def make(self, seed, threads):
        return LogisticRegression(max_iter=500, random_state=seed)

    def predict_proba(self, X, *, meter):
        for _ in range(X.shape[0]):
            meter.record("stub-model-1", input_tokens=500, output_tokens=10)
        return super().predict_proba(X, meter=meter)


class NeedsToken(Method):
    name = "needs_token"

    def unavailable(self):
        return "needs a licence token (NEEDS_TOKEN_KEY is not set)"


class Broken(SklearnMethod):
    name = "broken"

    def make(self, seed, threads):
        raise RuntimeError("this method always fails")


for m in (PricedStub, NeedsToken, Broken):
    register(m)
'''

PRICES = {
    "currency": "USD",
    "models": {
        "stub-model-1": {
            "input_per_1k": 0.003,
            "output_per_1k": 0.015,
            "read_on": "2026-10-01",
            "source_url": "https://example.com/pricing",
        }
    },
}


class Bench(Project):
    """A project frozen and registered, ready for a grid."""

    def build(self, **grid):
        from accbench.channels import write_manifest
        from accbench.fixtures import freeze

        self.write()
        (self.root / "channels.yaml").write_text(yaml.safe_dump(REGISTRY, sort_keys=False))
        (self.root / "my_methods.py").write_text(METHODS_MODULE)
        (self.root / "prices.yaml").write_text(yaml.safe_dump(PRICES, sort_keys=False))
        self.fixture = freeze(self.spec_path, out_dir=self.out)
        write_manifest(
            self.root / "channels.yaml", self.fixture, self.root / "channel_manifest.json"
        )
        data = {
            "construct": "construct.yaml",
            "fixture": self.fixture.path.relative_to(self.root).as_posix(),
            "registry": "channels.yaml",
            "channel_manifest": "channel_manifest.json",
            "methods": ["majority", "logreg"],
            "methods_module": "my_methods.py",
            "fit_seeds": [0],
            "threads": 1,
            "out": "runs",
        }
        data.update(grid)
        self.grid_data = data
        self.write_grid()
        return self

    def write_grid(self):
        (self.root / "grid.yaml").write_text(yaml.safe_dump(self.grid_data, sort_keys=False))

    @property
    def grid(self):
        return self.root / "grid.yaml"

    @property
    def runs(self):
        return self.root / "runs"


@pytest.fixture(scope="session")
def suite_record(tmp_path_factory):
    """One gate-sized null-suite record for the built-in rules, shared by every bench
    (bench projects hold the same code files, so they share a code state)."""
    from accbench.provenance import code_state
    from accbench.rules import ensure_null_suite

    ref = Bench(tmp_path_factory.mktemp("ref") / "bench").build()
    path = ref.root / "null_suite.json"
    ensure_null_suite(
        ["difference_clears_floor", "k_of_n_cells", "difference_of_drops"],
        path,
        code=code_state([ref.root]),
    )
    return path.read_text()


@pytest.fixture
def bench(tmp_path, suite_record):
    b = Bench(tmp_path / "bench")
    b.root.mkdir(parents=True, exist_ok=True)
    (b.root / "null_suite.json").write_text(suite_record)
    return b
