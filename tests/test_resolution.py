import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from accbench.cli import main
from accbench.errors import CodeStateMismatch, StaleInput
from accbench.provenance import code_state
from accbench.resampling import (
    bootstrap_metric,
    cluster_index,
    difference_in_differences,
    paired_difference,
)
from accbench.resolution import ResolutionError, floors, read_resolution, resolve
from accbench.runner import IncompleteGrid, run_grid

# ---------------------------------------------------------------- resampling


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = y * 0.3 + rng.normal(0, 0.5, n)
    return y, s


def test_cluster_index_groups_rows():
    groups = cluster_index(np.array(["b", "a", "b", "c", "a"]))
    assert [sorted(g.tolist()) for g in groups] == [[1, 4], [0, 2], [3]]


def test_row_bootstrap_on_repeated_trials_is_too_narrow():
    # Section 11: a pooled bootstrap counted each trial once per seed, making intervals
    # too narrow. Five copies of each trial: resampling rows narrows the interval by
    # about sqrt(5); resampling trials does not.
    y, s = _data()
    reps = 5
    y5, s5 = np.tile(y, reps), np.tile(s, reps)
    trial = np.tile(np.arange(len(y)), reps)
    by_row = bootstrap_metric(roc_auc_score, y5, s5, np.arange(len(y5)), n_boot=400)
    by_trial = bootstrap_metric(roc_auc_score, y5, s5, trial, n_boot=400)
    single = bootstrap_metric(roc_auc_score, y, s, np.arange(len(y)), n_boot=400)
    assert by_trial.half_width / by_row.half_width > 1.8
    assert by_trial.half_width == pytest.approx(single.half_width, rel=0.25)


def test_bootstrap_is_seeded():
    y, s = _data()
    a = bootstrap_metric(roc_auc_score, y, s, np.arange(len(y)), n_boot=200, seed=3)
    b = bootstrap_metric(roc_auc_score, y, s, np.arange(len(y)), n_boot=200, seed=3)
    assert a == b


def test_paired_difference():
    y, s = _data()
    same = paired_difference(roc_auc_score, y, s, s, np.arange(len(y)), n_boot=200)
    assert same.point == 0 and same.lo == 0 and same.hi == 0
    noise = np.random.default_rng(1).normal(0, 2, len(y))
    better = paired_difference(roc_auc_score, y, s, noise, np.arange(len(y)), n_boot=300)
    assert better.point > 0 and better.excludes_zero()


def test_difference_in_differences_null():
    # Model and reference drop by the same amount in expectation: across repeated
    # datasets the interval on the difference of drops should rarely exclude zero.
    excluded = 0
    for rep in range(12):
        rng = np.random.default_rng(100 + rep)
        groups = []
        for _ in range(2):
            y = rng.integers(0, 2, 300)
            a = y * 0.3 + rng.normal(0, 0.5, 300)
            b = a + rng.normal(0, 0.3, 300)
            groups.append((y, a, b, np.arange(300)))
        iv = difference_in_differences(roc_auc_score, *groups, n_boot=200, seed=rep)
        excluded += iv.excludes_zero()
    assert excluded <= 3


def test_one_class_resamples_are_counted_not_crashed():
    y = np.array([0] * 30 + [1])
    s = np.linspace(0, 1, 31)
    iv = bootstrap_metric(roc_auc_score, y, s, np.arange(31), n_boot=200)
    assert iv.n_failed > 0 and not np.isnan(iv.half_width)


# ---------------------------------------------------------------- the resolution table


def _full(bench, **grid):
    defaults = dict(
        methods=["majority", "logreg", "hist_gbm"],
        split_keys=["split-0", "split-1", "split-2"],
        fit_seeds=[0, 1],
    )
    defaults.update(grid)
    bench.build(**defaults)
    run_grid(bench.grid)
    return bench


def test_floor_is_the_largest_measured_term(bench):
    _full(bench)
    table = resolve(bench.grid, n_boot=150)
    assert [r["cell"] for r in table["cells"]] == [
        "mortality/Phase1",
        "mortality/Phase2",
        "mortality/Phase3",
    ]
    assert table["terms_measured"] == {"split": True, "fit": True, "test": True}
    for row in table["cells"]:
        terms = row["terms"]
        assert row["floor"] == max(terms.values())
        assert terms[row["source"]] == row["floor"]
        assert row["methods"] == ["hist_gbm", "logreg"]  # majority left out by default
        assert row["n_test_clusters"] == row["n_test"]
        assert row["resolves_threshold"] == (row["floor"] <= 0.05)
    assert table["underpowered"] == [r["cell"] for r in table["cells"] if r["floor"] > 0.05]
    assert set(floors(table)) == {r["cell"] for r in table["cells"]}


def test_unmeasured_terms_are_none_not_zero(bench):
    _full(bench, split_keys=["split-0"], fit_seeds=[0])
    table = resolve(bench.grid, n_boot=100)
    assert table["terms_measured"] == {"split": False, "fit": False, "test": True}
    for row in table["cells"]:
        assert row["terms"]["split"] is None and row["terms"]["fit"] is None
        assert row["source"] == "test"


def test_logreg_fit_term_is_zero_when_only_the_seed_changes(bench):
    # A deterministic solver does not move with the fit seed; the split term carries it.
    _full(bench, methods=["logreg"])
    table = resolve(bench.grid, n_boot=100)
    for row in table["cells"]:
        assert row["per_method"]["logreg"]["fit"] == pytest.approx(0, abs=1e-12)
        assert row["per_method"]["logreg"]["split"] > 0


def test_refuses_an_incomplete_grid(bench):
    _full(bench, methods=["logreg"], split_keys=["split-0"], fit_seeds=[0])
    next(bench.runs.rglob("*Phase2*.json")).unlink()
    with pytest.raises(IncompleteGrid):
        resolve(bench.grid, n_boot=50)


def test_refuses_records_from_other_code(bench):
    _full(bench, methods=["logreg"], split_keys=["split-0"], fit_seeds=[0])
    (bench.root / "my_methods.py").write_text(
        (bench.root / "my_methods.py").read_text() + "\n# edited\n"
    )
    with pytest.raises(IncompleteGrid, match="stale"):
        resolve(bench.grid, n_boot=50)


def test_stale_floor_table_is_refused_end_to_end(bench):
    # Section 10 / T1: the floor table was computed, then the records behind it were
    # re-run after a repair, and fifteen scripts kept reading the old table.
    _full(bench, methods=["logreg"], split_keys=["split-0"], fit_seeds=[0])
    resolve(bench.grid, n_boot=50)
    path = bench.root / "resolution.json"
    assert read_resolution(path).data["cells"]

    (bench.root / "my_methods.py").write_text(
        (bench.root / "my_methods.py").read_text() + "\n# repaired\n"
    )
    run_grid(bench.grid, on_stale="rerun")  # the re-run after the repair rewrites the records
    with pytest.raises(StaleInput, match="'runs'"):
        read_resolution(path)


def test_reader_with_new_code_refuses_the_table(bench):
    _full(bench, methods=["logreg"], split_keys=["split-0"], fit_seeds=[0])
    resolve(bench.grid, n_boot=50)
    (bench.root / "analysis.py").write_text("x = 1\n")
    with pytest.raises(CodeStateMismatch):
        read_resolution(bench.root / "resolution.json", expected_code=code_state([bench.root]))


def test_skipped_methods_are_listed(bench):
    _full(bench, methods=["logreg", "needs_token"], split_keys=["split-0"], fit_seeds=[0])
    table = resolve(bench.grid, methods=["logreg", "needs_token"], n_boot=50)
    assert all(r["skipped_methods"] == ["needs_token"] for r in table["cells"])
    assert all(r["methods"] == ["logreg"] for r in table["cells"])


def test_unknown_method(bench):
    _full(bench, methods=["logreg"], split_keys=["split-0"], fit_seeds=[0])
    with pytest.raises(ResolutionError, match="not in the grid"):
        resolve(bench.grid, methods=["xgboost"], n_boot=50)


def test_cli_resolve(bench, capsys):
    _full(bench, methods=["logreg"], split_keys=["split-0", "split-1"], fit_seeds=[0])
    assert main(["resolve", str(bench.grid), "--n-boot", "50"]) == 0
    out = capsys.readouterr().out
    assert "mortality/Phase1" in out and "floor" in out
    assert (bench.root / "resolution.json").exists()
