import copy

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from accbench.channels import Featurizer, Registry, load_registry, write_manifest
from accbench.cli import main
from accbench.errors import ChannelError, StaleInput, TrainOnlyViolation, UnregisteredColumn
from accbench.fixtures import freeze
from accbench.provenance import read_artifact

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


def _registry(mutate=None):
    data = copy.deepcopy(REGISTRY)
    if mutate:
        mutate(data)
    return Registry.from_dict(data)


def _problems(mutate):
    with pytest.raises(ChannelError) as err:
        _registry(mutate)
    return err.value.problems


# ---------------------------------------------------------------- the registry


def test_valid_registry_with_defaults():
    reg = _registry()
    assert reg.channels["tabular"].params == {
        "impute": "median",
        "scale": True,
        "missing_indicator": False,
    }
    assert reg.owner("age") == "tabular"
    assert reg.owner("died") == "ignore"
    assert reg.owner("city") is None


def test_one_channel_per_column():
    problems = _problems(lambda d: d["channels"]["tabular"]["columns"].append("phase"))
    assert any("one channel per column" in p for p in problems)
    problems = _problems(lambda d: d["channels"]["tabular"]["columns"].append("died"))
    assert any("'died' is in both ignore and tabular" in p for p in problems)


def test_categorical_encoding_has_no_default():
    problems = _problems(lambda d: d["channels"]["phase"].pop("encoding"))
    assert "channels.phase.encoding: required for a categorical channel (no default)" in problems


def test_unknown_kind_and_setting():
    problems = _problems(lambda d: d["channels"]["tabular"].update(kind="auto"))
    assert any("kind" in p for p in problems)
    problems = _problems(lambda d: d["channels"]["summary"].update(encoding="target"))
    assert "channels.summary.encoding: not a setting of a text channel" in problems


def test_bad_setting_values():
    problems = _problems(lambda d: d["channels"]["tabular"].update(impute="drop", scale="yes"))
    assert len(problems) == 2


def test_ignored_column_needs_a_reason():
    problems = _problems(lambda d: d["ignore"].update(trial_id=""))
    assert problems == ["ignore.trial_id: give the reason the column is kept out"]


def test_load_registry_missing(tmp_path):
    with pytest.raises(ChannelError, match="not found"):
        load_registry(tmp_path / "channels.yaml")


# ---------------------------------------------------------------- no default branch


def test_unregistered_column_raises(project):
    table = project.table.assign(city="Boston")
    with pytest.raises(UnregisteredColumn, match="'city' is not registered"):
        _registry().check_table(list(table.columns))


def test_misnamed_column_is_caught_both_ways():
    # Section 8 / T7: the featurizer listed "brief_summary" where the data has
    # "brief_summary/textblock", and the real column fell into a default branch.
    reg = _registry(lambda d: d["channels"]["summary"].update(columns=["brief_summary"]))
    cols = ["trial_id", "phase", "age", "enrollment", "brief_summary/textblock", "died"]
    cols += ["registered_on", "results_posted_on"]
    with pytest.raises(UnregisteredColumn) as err:
        reg.check_table(cols)
    problems = err.value.problems
    assert any("'brief_summary/textblock' is not registered" in p for p in problems)
    assert any("registers 'brief_summary', which the table lacks" in p for p in problems)


def test_label_column_cannot_sit_in_a_channel(project):
    reg = _registry(
        lambda d: (
            d["ignore"].pop("died"),
            d["channels"]["tabular"]["columns"].append("died"),
        )
    )
    with pytest.raises(ChannelError, match="would leak"):
        reg.check_table(list(project.table.columns), label_columns=["died"])


def test_channels_scoped_to_tasks(project):
    reg = _registry(lambda d: d["channels"]["summary"].update(tasks=["dropout"]))
    cols = [c for c in project.table.columns if c != "summary"]
    reg.check_table(cols, task="mortality")  # summary is not expected for mortality
    with pytest.raises(UnregisteredColumn):
        reg.check_table(list(project.table.columns), task="mortality")


# ---------------------------------------------------------------- train-only fit


def _split(table, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(["train", "test"], len(table), p=[0.7, 0.3]), index=table.index)


def test_fit_refuses_non_training_rows(project):
    t = project.table
    with pytest.raises(TrainOnlyViolation, match="non-training"):
        Featurizer(_registry()).fit_transform(t, _split(t), t["died"])


def test_transform_before_fit_raises(project):
    with pytest.raises(ChannelError, match="before fit"):
        Featurizer(_registry()).transform(project.table)


def test_fit_and_transform(project):
    t = project.table
    split = _split(t)
    train, test = t[split == "train"], t[split == "test"]
    fz = Featurizer(_registry())
    xtr = fz.fit_transform(train, split[split == "train"], train["died"])
    xte = fz.transform(test)
    assert xtr.shape[1] == xte.shape[1] == len(fz.feature_names_)
    assert xtr.shape[0] == len(train) and xte.shape[0] == len(test)
    assert fz.feature_names_[0] == "tabular:age"
    assert any(n.startswith("phase:") for n in fz.feature_names_)
    assert any(n.startswith("summary:") for n in fz.feature_names_)


def test_transform_checks_the_table_too(project):
    t = project.table
    fz = Featurizer(_registry())
    fz.fit_transform(t, pd.Series("train", index=t.index), t["died"])
    with pytest.raises(UnregisteredColumn):
        fz.transform(t.assign(city="Boston"))


def test_channel_subset(project):
    t = project.table
    fz = Featurizer(_registry(), channels=["summary"])
    fz.fit_transform(t, pd.Series("train", index=t.index), t["died"])
    assert all(n.startswith("summary:") for n in fz.feature_names_)
    with pytest.raises(ChannelError, match="not in the registry"):
        Featurizer(_registry(), channels=["nope"])


def test_unseen_category_at_test_is_harmless(project):
    t = project.table
    fz = Featurizer(_registry(), channels=["phase"])
    fz.fit_transform(t, pd.Series("train", index=t.index), t["died"])
    x = fz.transform(t.head(3).assign(phase="Phase4")).toarray()
    assert (x == 0).all()


def test_multihot_and_missing_numeric(project):
    t = project.table.assign(codes=["I10;E11", "I10", None, "C50"] * 100)
    t.loc[:20, "age"] = np.nan
    reg = _registry(
        lambda d: d["channels"].update(
            codes={"kind": "multihot", "columns": ["codes"]},
            tabular={
                "kind": "numeric",
                "columns": ["age", "enrollment"],
                "missing_indicator": True,
            },
        )
    )
    fz = Featurizer(reg, channels=["codes", "tabular"])
    x = fz.fit_transform(t, pd.Series("train", index=t.index), t["died"]).toarray()
    assert {"codes:I10", "codes:E11", "codes:C50"} <= set(fz.feature_names_)
    assert any("missingindicator" in n for n in fz.feature_names_)
    assert not np.isnan(x).any()


def test_leak_mechanism_and_the_cross_fitted_guard():
    # Section 8 / T8: a near-unique column, target-encoded on all training rows,
    # separates the classes in training and collapses to a constant at test.
    rng = np.random.default_rng(0)
    n = 1595
    y = rng.integers(0, 2, n)
    unique_text = pd.Series([f"summary {i}" for i in range(n)])
    train = np.arange(n) < 1200

    # The defect: per-value outcome rates fitted on the training rows themselves.
    rates = pd.Series(y[train]).groupby(unique_text[train].to_numpy()).mean()
    prior = y[train].mean()
    leaked_train = unique_text[train].map(rates).to_numpy()
    leaked_test = unique_text[~train].map(rates).fillna(prior).to_numpy()
    assert roc_auc_score(y[train], leaked_train) > 0.99
    assert np.unique(leaked_test).size == 1

    # The guard: registered as a target-encoded channel, the encoding is cross-fitted.
    reg = Registry.from_dict(
        {
            "channels": {"s": {"kind": "categorical", "columns": ["s"], "encoding": "target"}},
            "ignore": {},
        }
    )
    frame = pd.DataFrame({"s": unique_text})
    fz = Featurizer(reg)
    x = fz.fit_transform(frame[train], pd.Series("train", index=frame.index[train]), y[train])
    assert roc_auc_score(y[train], x.toarray()[:, 0]) < 0.6


# ---------------------------------------------------------------- the manifest


def _write_registry(project, data=None):
    path = project.root / "channels.yaml"
    path.write_text(yaml.safe_dump(data or REGISTRY, sort_keys=False))
    return path


def test_manifest_records_each_columns_path(project):
    project.write()
    fx = freeze(project.spec_path, out_dir=project.out)
    reg = _write_registry(project)
    out = project.root / "channel_manifest.json"
    write_manifest(reg, fx, out)
    art = read_artifact(out)
    cols = art.data["tasks"]["mortality"]["columns"]
    assert cols["age"] == "tabular" and cols["died"] == "ignore"
    assert art.data["fixture_hash"] == fx.hash


def test_manifest_goes_stale_when_registry_changes(project):
    project.write()
    fx = freeze(project.spec_path, out_dir=project.out)
    reg = _write_registry(project)
    out = project.root / "channel_manifest.json"
    write_manifest(reg, fx, out)
    data = copy.deepcopy(REGISTRY)
    data["channels"]["tabular"]["scale"] = False
    _write_registry(project, data)
    with pytest.raises(StaleInput):
        read_artifact(out)


def test_manifest_refuses_label_in_a_channel(project):
    project.write()
    fx = freeze(project.spec_path, out_dir=project.out)
    data = copy.deepcopy(REGISTRY)
    data["ignore"].pop("died")
    data["channels"]["tabular"]["columns"].append("died")
    with pytest.raises(ChannelError, match="would leak"):
        write_manifest(_write_registry(project, data), fx, project.root / "m.json")


def test_fixture_rows_align(project):
    project.write()
    fx = freeze(project.spec_path, out_dir=project.out)
    a, f = fx.rows("mortality")
    assert (a["id"].to_numpy() == f["trial_id"].to_numpy()).all()
    assert (a["label"].to_numpy() == f["died"].to_numpy()).all()


def test_cli_channels_check(project, capsys):
    project.write()
    fx = freeze(project.spec_path, out_dir=project.out)
    reg = _write_registry(project)
    out = project.root / "channel_manifest.json"
    assert main(["channels", "check", str(reg), "--fixture", str(fx.path), "--out", str(out)]) == 0
    assert "every column registered" in capsys.readouterr().out
    project.table.assign(city="Boston").to_csv(project.root / "data" / "mortality.csv", index=False)
    fx2 = freeze(project.spec_path, out_dir=project.out)
    assert main(["channels", "check", str(reg), "--fixture", str(fx2.path), "--out", str(out)]) == 1
    assert "'city' is not registered" in capsys.readouterr().err
