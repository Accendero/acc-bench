import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from accbench.cli import main
from accbench.costs import Meter, PriceError, PriceTable, UnpricedModel
from accbench.errors import StaleInput
from accbench.metrics import score, tie_share
from accbench.provenance import read_artifact
from accbench.runner import (
    GridError,
    IncompleteGrid,
    coverage,
    load_grid,
    prepare,
    require_complete,
    run_grid,
)

CELLS = ("mortality/Phase1", "mortality/Phase2", "mortality/Phase3")


def _records(bench):
    return sorted(p for p in bench.runs.rglob("*.json"))


# ---------------------------------------------------------------- a complete grid


def test_every_planned_run_leaves_a_record(bench):
    bench.build(methods=["majority", "logreg", "tfidf_logreg"], fit_seeds=[0, 1])
    counts = run_grid(bench.grid)
    assert counts == {"resumed": 0, "ok": 18, "skipped": 0, "error": 0}
    assert len(_records(bench)) == 18  # 3 cells x 3 methods x 2 seeds
    cov = coverage(bench.grid)
    assert cov["planned"] == 18 and cov["complete"]
    assert sum(cov["counts"].values()) == cov["planned"]
    require_complete(cov)


def test_record_contents(bench):
    bench.build()
    run_grid(bench.grid)
    path = bench.runs / "mortality" / "mortality--Phase1__logreg__split-0__s0.json"
    art = read_artifact(path)
    r = art.data
    assert r["status"] == "ok" and r["metric"] == "pr_auc"
    assert r["threads"] == 1 and isinstance(r["threadpools"], list)
    assert set(r["n"]) == {"train", "validation", "test"}
    assert 0 <= r["results"]["test"]["score"] <= 1
    assert r["channels"] == ["tabular", "phase"]  # logreg reads no text channel
    assert r["cost"]["total"] == 0
    assert art.code.digest  # stamped with the working-tree code state
    assert set(art.inputs) == {"fixture", "channel_manifest", "registry", "predictions"}

    pred = pd.read_csv(
        bench.runs / "mortality" / "mortality--Phase1__logreg__split-0__s0.predictions.csv"
    )
    assert set(pred["split"]) == {"validation", "test"}
    assert len(pred) == r["n"]["validation"] + r["n"]["test"]
    assert pred["cluster"].notna().all()


def test_same_seed_same_result(bench):
    bench.build(methods=["hist_gbm"], fit_seeds=[0])
    run_grid(bench.grid)
    first = {p.name: read_artifact(p).data["results"] for p in _records(bench)}
    for p in _records(bench):
        p.unlink()
    run_grid(bench.grid)
    second = {p.name: read_artifact(p).data["results"] for p in _records(bench)}
    assert first == second


# ---------------------------------------------------------------- resume


def test_resume_skips_finished_runs(bench):
    bench.build()
    run_grid(bench.grid)
    mtimes = {p: p.stat().st_mtime_ns for p in _records(bench)}
    assert run_grid(bench.grid) == {"resumed": 6, "ok": 0, "skipped": 0, "error": 0}
    assert {p: p.stat().st_mtime_ns for p in _records(bench)} == mtimes


def test_interrupted_grid_resumes_where_it_stopped(bench):
    bench.build()
    run_grid(bench.grid)
    _records(bench)[2].unlink()
    assert run_grid(bench.grid)["ok"] == 1


def test_stale_records_are_refused_not_mixed(bench):
    # Section 9: records from before a repair must not be resumed under the repaired code.
    bench.build()
    run_grid(bench.grid)
    (bench.root / "my_methods.py").write_text(
        (bench.root / "my_methods.py").read_text() + "\n# repaired\n"
    )
    with pytest.raises(GridError, match="stale"):
        run_grid(bench.grid)
    cov = coverage(bench.grid)
    assert cov["counts"]["stale"] == 6 and not cov["complete"]
    assert run_grid(bench.grid, on_stale="rerun")["ok"] == 6
    assert coverage(bench.grid)["complete"]


# ---------------------------------------------------------------- skips and errors


def test_skip_is_recorded_with_its_reason(bench):
    # Section 9: TabPFN needed a licence token, so the runner recorded a clean skip.
    bench.build(methods=["logreg", "needs_token"])
    counts = run_grid(bench.grid)
    assert counts["skipped"] == 3 and counts["ok"] == 3
    cov = coverage(bench.grid)
    assert cov["complete"] and cov["counts"]["skipped"] == 3
    skipped = [r for r in cov["runs"] if r["status"] == "skipped"]
    assert all("licence token" in r["reason"] for r in skipped)


def test_method_without_a_matching_channel_is_skipped(bench):
    reg = copy.deepcopy(yaml.safe_load((bench.build().root / "channels.yaml").read_text()))
    assert reg  # built
    bench.grid_data["methods"] = ["tfidf_logreg"]
    bench.grid_data["tasks"] = ["mortality"]
    bench.write_grid()
    reg["channels"].pop("summary")
    reg["ignore"]["summary"] = "left out for this test"
    (bench.root / "channels.yaml").write_text(yaml.safe_dump(reg))
    from accbench.channels import write_manifest

    write_manifest(
        bench.root / "channels.yaml", bench.fixture, bench.root / "channel_manifest.json"
    )
    run_grid(bench.grid)
    r = read_artifact(_records(bench)[0]).data
    assert r["status"] == "skipped" and "no registered channel" in r["reason"]


def test_failure_is_recorded_and_the_grid_carries_on(bench):
    bench.build(methods=["broken", "logreg"])
    counts = run_grid(bench.grid)
    assert counts["error"] == 3 and counts["ok"] == 3
    rec = read_artifact(bench.runs / "mortality" / "mortality--Phase2__broken__split-0__s0.json")
    assert rec.data["status"] == "error"
    assert "always fails" in rec.data["reason"] and "Traceback" in rec.data["traceback"]
    cov = coverage(bench.grid)
    assert not cov["complete"]
    with pytest.raises(IncompleteGrid, match="average each method over different cells"):
        require_complete(cov)


def test_missing_runs_make_the_grid_incomplete(bench):
    # Section 3 / T4: a method ranked on the cells it had finished.
    bench.build()
    run_grid(bench.grid)
    for p in list(bench.runs.rglob("*logreg*")):
        if "Phase3" in p.name:
            p.unlink()
    cov = coverage(bench.grid)
    assert cov["counts"]["missing"] == 1 and cov["by_method"]["logreg"]["missing"] == 1
    with pytest.raises(IncompleteGrid):
        require_complete(cov)


def test_errors_rerun_on_resume(bench):
    bench.build(methods=["broken"])
    run_grid(bench.grid)
    assert run_grid(bench.grid)["error"] == 3


# ---------------------------------------------------------------- cost


def test_cost_is_metered_against_the_pinned_table(bench):
    bench.build(methods=["priced_stub"], prices="prices.yaml")
    run_grid(bench.grid)
    for p in _records(bench):
        r = read_artifact(p).data
        n_eval = r["n"]["validation"] + r["n"]["test"]
        call = r["cost"]["by_model"]["stub-model-1"]
        assert call["calls"] == n_eval
        assert r["cost"]["total"] == pytest.approx(n_eval * (0.5 * 0.003 + 0.01 * 0.015))
        assert len(r["cost"]["price_table_sha256"]) == 64


def test_billable_call_without_a_price_table_fails_closed(bench):
    bench.build(methods=["priced_stub"])
    assert run_grid(bench.grid)["error"] == 3
    r = read_artifact(_records(bench)[0]).data
    assert "UnpricedModel" in r["reason"]


def test_price_table_validation():
    with pytest.raises(PriceError, match="source_url"):
        PriceTable.from_dict(
            {"models": {"m": {"input_per_1k": 1, "output_per_1k": 1, "read_on": "2026-01-01"}}}
        )
    table = PriceTable.from_dict(
        {
            "models": {
                "m": {
                    "input_per_1k": 1.0,
                    "output_per_1k": 2.0,
                    "read_on": "2026-01-01",
                    "source_url": "https://x",
                }
            }
        }
    )
    meter = Meter(table)
    assert meter.record("m", input_tokens=1000, output_tokens=500) == pytest.approx(2.0)
    with pytest.raises(UnpricedModel):
        meter.record("other", input_tokens=1, output_tokens=1)


# ---------------------------------------------------------------- what the grid checks first


def test_grid_refuses_an_amended_construct_without_a_new_fixture(bench):
    bench.build()
    c = yaml.safe_load((bench.root / "construct.yaml").read_text())
    c["version"] = 2
    c["amendments"] = [{"version": 2, "date": "2026-11-01", "reason": "Added a reported number."}]
    (bench.root / "construct.yaml").write_text(yaml.safe_dump(c))
    with pytest.raises(GridError, match="freeze it again"):
        run_grid(bench.grid)


def test_grid_refuses_a_stale_channel_manifest(bench):
    bench.build()
    reg = yaml.safe_load((bench.root / "channels.yaml").read_text())
    reg["channels"]["tabular"]["scale"] = False
    (bench.root / "channels.yaml").write_text(yaml.safe_dump(reg))
    with pytest.raises(StaleInput):
        run_grid(bench.grid)


def test_grid_validation(bench):
    bench.build(methods=["nope"])
    with pytest.raises(GridError, match="unknown method"):
        load_grid(bench.grid)
    bench.grid_data.update(methods=["logreg"], split_keys=["never-frozen"])
    bench.write_grid()
    with pytest.raises(GridError, match="were not frozen"):
        prepare(load_grid(bench.grid))
    bench.grid_data.update(split_keys=["split-0"], threads=0, colour="blue")
    bench.write_grid()
    with pytest.raises(GridError, match="colour: unknown key; threads"):
        load_grid(bench.grid)


def test_split_keys_and_seeds_multiply_the_plan(bench):
    bench.build(split_keys=["split-0", "split-1"], fit_seeds=[0, 1, 2])
    ctx = prepare(load_grid(bench.grid))
    assert len(ctx.plan()) == 3 * 2 * 2 * 3


# ---------------------------------------------------------------- multiclass


def test_multiclass_task(bench):
    bench.root.mkdir(parents=True, exist_ok=True)
    (bench.root / "labels.py").write_text(
        "import pandas as pd\n"
        "def dropout(t):\n"
        "    return pd.cut(t['enrollment'], [0, 500, 1200, 10**6], labels=False)\n"
    )
    bench.spec["tasks"]["dropout"] = {
        "source": "data/mortality.csv",
        "type": "multiclass",
        "label_rule": "labels:dropout",
    }
    bench.build(tasks=["dropout"])
    assert run_grid(bench.grid)["ok"] == 2
    rec = read_artifact(bench.runs / "dropout" / "dropout__logreg__split-0__s0.json").data
    assert rec["results"]["test"]["score"] > 0.5  # enrollment is a feature
    pred = pd.read_csv(bench.runs / "dropout" / "dropout__logreg__split-0__s0.predictions.csv")
    assert {"p_0", "p_1", "p_2"} <= set(pred.columns)


# ---------------------------------------------------------------- metrics


def test_metrics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert score("roc_auc", y, s) == pytest.approx(0.75)
    assert score("pr_auc", y, s) == pytest.approx(0.8333, abs=1e-3)
    assert score("accuracy", y, s) == pytest.approx(0.75)
    assert score("brier", y, np.array([0, 0, 1, 1.0])) == 0
    with pytest.raises(ValueError):
        score("auroc", y, s)


def test_tie_share():
    assert tie_share(np.array([0.1, 0.2, 0.3])) == 0
    assert tie_share(np.array([0.5, 0.5, 0.5, 0.9])) == pytest.approx(0.75)


# ---------------------------------------------------------------- CLI


def test_cli_run_and_coverage(bench, capsys):
    bench.build(methods=["logreg", "broken"])
    assert main(["run", str(bench.grid)]) == 1  # errors make the grid incomplete
    out = capsys.readouterr().out
    assert "coverage: 6 planned = ok 3 + skipped 0 + error 3" in out and "INCOMPLETE" in out
    bench.grid_data["methods"] = ["logreg"]
    bench.write_grid()
    assert main(["coverage", str(bench.grid)]) == 0


def test_fast_metrics_equal_scikit_learn():
    from sklearn import metrics as skm

    from accbench.metrics import average_precision, roc_auc

    rng = np.random.default_rng(0)
    for i in range(300):
        n = int(rng.integers(5, 300))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        s = rng.normal(size=n)
        if i % 2:
            s = np.round(s, 1)  # ties
        if i % 7 == 0:
            s = np.zeros(n)  # all tied
        assert average_precision(y, s) == pytest.approx(
            skm.average_precision_score(y, s), abs=1e-12
        )
        assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)
    with pytest.raises(ValueError):
        roc_auc(np.zeros(4, dtype=int), np.arange(4.0))


def test_grid_may_name_the_fixture_spec(bench):
    bench.build(fixture="fixtures.yaml")
    assert run_grid(bench.grid)["ok"] == 6


def test_grid_naming_an_unfrozen_spec_is_refused(bench):
    from accbench.fixtures import FixtureError

    bench.build(fixture="fixtures.yaml")
    bench.table.loc[0, "age"] = 99.0  # the data moved since the freeze
    bench.table.to_csv(bench.root / "data" / "mortality.csv", index=False)
    with pytest.raises(FixtureError, match="run acc-bench fixtures freeze first"):
        run_grid(bench.grid)
