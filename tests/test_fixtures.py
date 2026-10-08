import json

import pandas as pd
import pytest

from accbench.cli import main
from accbench.errors import ConstructError, StaleInput
from accbench.fixtures import (
    FixtureError,
    assign_arms,
    assign_splits,
    freeze,
    load_fixture,
    unit_hash,
)
from accbench.provenance import code_state


def _freeze(project):
    project.write()
    return freeze(project.spec_path, out_dir=project.out, code=code_state([project.root]))


# ---------------------------------------------------------------- freezing


def test_freeze_writes_a_content_addressed_fixture(project):
    fx = _freeze(project)
    assert fx.path.name == fx.hash[:16]
    assert fx.tasks == ("mortality",)
    assert fx.cells("mortality") == ("mortality/Phase1", "mortality/Phase2", "mortality/Phase3")
    assert fx.split_keys == ("split-0", "split-1", "split-2")
    a = fx.assignments("mortality")
    assert list(a.columns) == [
        "id",
        "cluster",
        "cell",
        "label",
        "split__split-0",
        "split__split-1",
        "split__split-2",
        "arm",
    ]
    assert len(a) == 400


def test_manifest_says_what_was_frozen_and_when(project):
    # Acceptance (section 7): for every input, a reader can say what was frozen and when.
    fx = _freeze(project)
    m = fx.manifest.data
    assert m["frozen_at"]
    assert m["construct"]["version"] == 1
    t = m["tasks"]["mortality"]
    assert t["label_rule"] == {"kind": "column", "column": "died"}
    assert t["source"]["sha256"]
    assert sum(t["split_counts"]["mortality/Phase1"]["split-0"].values()) > 0
    assert set(fx.manifest.inputs) == {"spec", "source:mortality"}


def test_freezing_twice_gives_the_same_fixture(project):
    first = _freeze(project)
    second = freeze(project.spec_path, out_dir=project.out, code=code_state([project.root]))
    assert first.hash == second.hash
    assert first.frozen_at == second.frozen_at
    assert len(list(project.out.iterdir())) == 1


def test_a_changed_spec_gives_a_new_fixture(project):
    first = _freeze(project)
    project.spec["split"]["keys"] = ["split-0", "split-1"]
    second = _freeze(project)
    assert first.hash != second.hash
    assert len(list(project.out.iterdir())) == 2


def test_cli_freeze_and_show(project, capsys):
    project.write()
    assert main(["fixtures", "freeze", str(project.spec_path), "--out", str(project.out)]) == 0
    assert "frozen" in capsys.readouterr().out
    assert main(["fixtures", "show", str(next(project.out.iterdir()))]) == 0
    assert "frozen" in capsys.readouterr().out


# ---------------------------------------------------------------- the construct guard


def test_freeze_refuses_without_a_construct(project):
    project.write()
    (project.root / "construct.yaml").unlink()
    with pytest.raises(ConstructError, match="state the construct before freezing"):
        freeze(project.spec_path, out_dir=project.out)
    assert not project.out.exists()


def test_freeze_refuses_an_invalid_construct(project):
    del project.construct["decision"]
    project.write()
    with pytest.raises(ConstructError, match="decision: missing"):
        freeze(project.spec_path, out_dir=project.out)


def test_freeze_only_what_the_construct_names(project):
    project.spec["tasks"]["adverse_events"] = dict(project.spec["tasks"]["mortality"])
    project.write()
    with pytest.raises(FixtureError, match="not in the construct statement"):
        freeze(project.spec_path, out_dir=project.out)


def test_spec_problems_are_reported_together(project):
    project.spec["split"]["validation"] = {"fraction": 1.5}
    project.spec["tasks"]["mortality"]["type"] = "regression"
    project.spec["extra"] = 1
    project.write()
    with pytest.raises(FixtureError) as err:
        freeze(project.spec_path, out_dir=project.out)
    msg = str(err.value)
    assert "split.validation" in msg and "type" in msg and "extra: unknown key" in msg


# ---------------------------------------------------------------- frozen means frozen


def test_editing_a_frozen_fixture_is_refused(project):
    fx = _freeze(project)
    f = fx.path / "mortality.csv"
    f.write_text(f.read_text().replace(",train,", ",test,", 1))
    with pytest.raises(StaleInput, match="modified after it was frozen"):
        load_fixture(fx.path)


def test_changed_source_data_is_refused(project):
    fx = _freeze(project)
    project.table.loc[0, "died"] = 1 - project.table.loc[0, "died"]
    project.table.to_csv(project.root / "data" / "mortality.csv", index=False)
    with pytest.raises(StaleInput, match="source:mortality"):
        load_fixture(fx.path)


def test_not_a_fixture(tmp_path):
    with pytest.raises(FixtureError, match="not a frozen fixture"):
        load_fixture(tmp_path)


# ---------------------------------------------------------------- keyed splits


def test_split_does_not_depend_on_row_order(project):
    a = _freeze(project).assignments("mortality").set_index("id")
    project.table = project.table.sample(frac=1, random_state=3)
    b = (
        freeze(project.write().spec_path, out_dir=project.root / "other")
        .assignments("mortality")
        .set_index("id")
    )
    pd.testing.assert_frame_equal(a, b.loc[a.index])


def test_adding_rows_does_not_move_existing_ones(project):
    a = _freeze(project).assignments("mortality").set_index("id")
    from conftest import make_trials

    extra = make_trials(50, seed=9)
    extra["trial_id"] = [f"NEW{i:05d}" for i in range(50)]
    project.table = pd.concat([project.table, extra])
    b = (
        freeze(project.write().spec_path, out_dir=project.root / "other")
        .assignments("mortality")
        .set_index("id")
    )
    cols = [c for c in a.columns if c.startswith("split__")]
    pd.testing.assert_frame_equal(a[cols], b.loc[a.index, cols])


def test_test_set_is_fixed_across_split_keys():
    # Section 7: only the train/validation split is redrawn; the test set stays fixed.
    clusters = pd.Series([f"c{i}" for i in range(2000)])
    s = assign_splits(
        clusters, keys=["k0", "k1"], test={"fraction": 0.2, "key": "t"}, validation_fraction=0.25
    )
    assert ((s["k0"] == "test") == (s["k1"] == "test")).all()
    assert (s["k0"] != s["k1"]).any()  # a redraw under a new key moves validation rows
    test_share = (s["k0"] == "test").mean()
    val_share = (s["k0"] == "validation").mean()
    assert 0.17 < test_share < 0.23
    assert 0.17 < val_share < 0.23  # 0.25 of the remaining 0.8


def test_rows_of_one_cluster_share_a_split():
    clusters = pd.Series(["a", "a", "b", "b", "c", "c"] * 50 + [f"x{i}" for i in range(300)])
    s = assign_splits(
        clusters, keys=["k"], test={"fraction": 0.3, "key": "t"}, validation_fraction=0.3
    )["k"]
    assert s.groupby(clusters).nunique().max() == 1


def test_test_set_from_a_column(project):
    project.table["shipped_split"] = ["test" if i % 4 == 0 else "train" for i in range(400)]
    project.spec["split"]["test"] = {"column": "shipped_split", "value": "test"}
    a = _freeze(project).assignments("mortality").set_index("id")
    shipped = project.table.set_index("trial_id")["shipped_split"]
    assert ((a["split__split-0"] == "test") == (shipped.loc[a.index] == "test")).all()


def test_unit_hash_is_fixed():
    assert unit_hash("k", "NCT00000001") == unit_hash("k", "NCT00000001")
    assert 0 <= unit_hash("k", "x") < 1
    assert unit_hash("k", "x") != unit_hash("j", "x")


def test_duplicate_ids_are_refused(project):
    project.table = pd.concat([project.table, project.table.iloc[[5]]])
    project.write()
    with pytest.raises(FixtureError, match="more than once"):
        freeze(project.spec_path, out_dir=project.out)


# ---------------------------------------------------------------- one label rule


def test_label_rule_function(project):
    (project.root).mkdir(parents=True, exist_ok=True)
    (project.root / "labels.py").write_text(
        "def mortality(t):\n    return (t['died'] == 1).astype(int)\n"
    )
    project.spec["tasks"]["mortality"]["label_rule"] = "labels:mortality"
    project.spec["tasks"]["mortality"]["label_columns"] = ["died"]
    fx = _freeze(project)
    rule = fx.manifest.data["tasks"]["mortality"]["label_rule"]
    assert rule["kind"] == "function" and rule["ref"] == "labels:mortality"
    assert len(rule["source_sha256"]) == 64


def test_rows_without_a_label_are_counted_not_dropped_silently(project):
    project.table["died"] = project.table["died"].astype("Float64")
    project.table.loc[:9, "died"] = pd.NA
    fx = _freeze(project)
    t = fx.manifest.data["tasks"]["mortality"]
    assert t["excluded"] == {"label rule returned no label": 10}
    assert t["rows_in"] == 400 and t["rows_kept"] == 390


def test_label_map(project):
    project.table["died"] = project.table["died"].map({1: "yes", 0: "no"})
    project.spec["tasks"]["mortality"]["label_rule"] = {
        "column": "died",
        "map": {"yes": 1, "no": 0},
    }
    assert _freeze(project).assignments("mortality")["label"].isin([0, 1]).all()


def test_binary_task_refuses_other_labels(project):
    project.table.loc[0, "died"] = 2
    project.write()
    with pytest.raises(FixtureError, match="binary task"):
        freeze(project.spec_path, out_dir=project.out)


def test_a_rule_with_fallbacks_is_not_a_rule(project):
    # The campaign tried several candidate label columns and fell back to the last.
    project.spec["tasks"]["mortality"]["label_rule"] = {"column": ["died", "death"]}
    project.write()
    with pytest.raises(FixtureError, match="no fallbacks"):
        freeze(project.spec_path, out_dir=project.out)


# ---------------------------------------------------------------- cutoffs and arms


ARMS = {
    "A": {"cutoff": "model", "before": ["results_posted_on"]},
    "B": {"cutoff": "model", "before": ["registered_on"], "after": ["results_posted_on"]},
    "C": {"cutoff": "model", "after": ["registered_on"]},
}


def test_arms_against_a_cutoff(project):
    project.spec["cutoffs"] = {"model": "2022-01-01"}
    project.spec["arms"] = ARMS
    fx = _freeze(project)
    a = fx.assignments("mortality").set_index("id")
    t = project.table.set_index("trial_id")
    reg = pd.to_datetime(t["registered_on"])
    post = pd.to_datetime(t["results_posted_on"])
    cut = pd.Timestamp("2022-01-01")
    assert ((a["arm"] == "A") == (post.loc[a.index] < cut)).all()
    assert ((a["arm"] == "C") == (reg.loc[a.index] >= cut)).all()
    counts = fx.manifest.data["tasks"]["mortality"]["arm_counts"]
    assert sum(counts.values()) == 400 and set(counts) <= {"A", "B", "C", "(none)"}
    assert fx.manifest.data["cutoffs"] == {"model": "2022-01-01"}


def test_overlapping_arms_are_refused():
    table = pd.DataFrame({"d": ["2020-01-01", "2023-01-01"]})
    arms = {"X": {"cutoff": "m", "before": ["d"]}, "Y": {"cutoff": "m", "before": ["d"]}}
    with pytest.raises(FixtureError, match="overlap"):
        assign_arms(table, arms, {"m": "2022-01-01"})


def test_missing_dates_fall_in_no_arm():
    table = pd.DataFrame({"d": ["2020-01-01", None, "not a date"]})
    arm = assign_arms(table, {"X": {"cutoff": "m", "before": ["d"]}}, {"m": "2022-01-01"})
    assert arm.tolist() == ["X", None, None]


def test_arm_needs_its_columns(project):
    project.spec["cutoffs"] = {"model": "2022-01-01"}
    project.spec["arms"] = {"A": {"cutoff": "model", "before": ["no_such_column"]}}
    project.write()
    with pytest.raises(FixtureError, match="no_such_column"):
        freeze(project.spec_path, out_dir=project.out)


def test_arm_must_name_a_declared_cutoff(project):
    project.spec["arms"] = {"A": {"cutoff": "model", "before": ["registered_on"]}}
    project.write()
    with pytest.raises(FixtureError, match="arms.A.cutoff"):
        freeze(project.spec_path, out_dir=project.out)


def test_manifest_is_valid_json_with_stamp(project):
    fx = _freeze(project)
    raw = json.loads((fx.path / "manifest.json").read_text())
    assert "_provenance" in raw


def test_label_rules_from_two_projects_do_not_collide(tmp_path):
    from conftest import Project

    results = []
    for name, expr in (("p1", "t['died']"), ("p2", "1 - t['died']")):
        p = Project(tmp_path / name)
        p.root.mkdir(parents=True)
        (p.root / "labels.py").write_text(f"def mortality(t):\n    return {expr}\n")
        p.spec["tasks"]["mortality"]["label_rule"] = "labels:mortality"
        p.spec["tasks"]["mortality"]["label_columns"] = ["died"]
        fx = freeze(p.write().spec_path, out_dir=p.out)
        results.append(fx.assignments("mortality").set_index("id")["label"])
    assert (results[0] == 1 - results[1]).all()


def test_label_rule_that_cannot_be_imported(project):
    project.spec["tasks"]["mortality"]["label_rule"] = "no_such_module:f"
    project.spec["tasks"]["mortality"]["label_columns"] = ["died"]
    project.write()
    with pytest.raises(FixtureError, match="cannot import"):
        freeze(project.spec_path, out_dir=project.out)


def test_changed_source_data_gives_a_new_fixture(project):
    # A new column changes no assignment, but the frozen source hash must still move.
    first = _freeze(project)
    project.table = project.table.assign(city="Boston")
    second = _freeze(project)
    assert first.hash != second.hash


def test_function_label_rule_must_declare_what_it_reads(project):
    project.root.mkdir(parents=True, exist_ok=True)
    (project.root / "labels.py").write_text(
        "def mortality(t):\n    return ((t['died'] == 1) & (t['age'] > 0)).astype(int)\n"
    )
    project.spec["tasks"]["mortality"]["label_rule"] = "labels:mortality"
    project.write()
    with pytest.raises(FixtureError, match="must name the columns it reads"):
        freeze(project.spec_path, out_dir=project.out)
    project.spec["tasks"]["mortality"]["label_columns"] = ["died"]  # age is not declared
    project.write()
    with pytest.raises(FixtureError, match="does not name"):
        freeze(project.spec_path, out_dir=project.out)
    project.spec["tasks"]["mortality"]["label_columns"] = ["died", "age"]
    project.write()
    fx = freeze(project.spec_path, out_dir=project.out)
    assert fx.manifest.data["tasks"]["mortality"]["label_rule"]["columns"] == ["died", "age"]


def test_a_declared_label_column_cannot_sit_in_a_channel(project):
    from accbench.channels import Registry, write_manifest
    from conftest import REGISTRY

    project.root.mkdir(parents=True, exist_ok=True)
    (project.root / "labels.py").write_text(
        "def mortality(t):\n    return (t['enrollment'] > 1000).astype(int)\n"
    )
    project.spec["tasks"]["mortality"]["label_rule"] = "labels:mortality"
    project.spec["tasks"]["mortality"]["label_columns"] = ["enrollment"]
    fx = _freeze(project)
    import yaml

    (project.root / "channels.yaml").write_text(yaml.safe_dump(REGISTRY))
    assert Registry.from_dict(REGISTRY).owner("enrollment") == "tabular"
    from accbench.errors import ChannelError

    with pytest.raises(ChannelError, match="would leak"):
        write_manifest(project.root / "channels.yaml", fx, project.root / "m.json")
