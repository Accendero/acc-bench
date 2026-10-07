"""The defect gallery: each failure from the paper's campaign, rebuilt on synthetic data.

Every test reproduces one defect through the public API and shows the guard that stops
it. The paper section and test ID are in each docstring; docs/defect-gallery.md has the
same list as a table.
"""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from accbench import claims
from accbench.channels import write_manifest
from accbench.errors import StaleInput, UnregisteredColumn
from accbench.fixtures import freeze
from accbench.provenance import read_artifact
from accbench.resampling import bootstrap_metric
from accbench.resolution import read_resolution, resolve
from accbench.rules import (
    SyntheticContext,
    Verdict,
    _null_drops,
    decide,
    rule,
    run_null_suite,
)
from accbench.runner import GridError, IncompleteGrid, coverage, run_grid


def _small(bench, **grid):
    defaults = dict(methods=["logreg"], split_keys=["split-0"], fit_seeds=[0])
    defaults.update(grid)
    return bench.build(**defaults)


def _touch_code(bench, note="repaired"):
    f = bench.root / "my_methods.py"
    f.write_text(f.read_text() + f"\n# {note}\n")


# 1 ---------------------------------------------------------------- section 8, T7 and T8


def test_1_misnamed_column_cannot_reach_a_model(bench):
    """Campaign: the featurizer listed "brief_summary" where the data has
    "brief_summary/textblock"; the real column fell into a default categorical branch,
    was target-encoded, and leaked the label (T7, T8; section 8).
    Guard: the channel registry has no default branch, so both names are refused."""
    from conftest import REGISTRY

    bench.table = bench.table.rename(columns={"summary": "brief_summary/textblock"})
    bench.write()
    reg = {**REGISTRY, "channels": {**REGISTRY["channels"]}}
    reg["channels"]["summary"] = {"kind": "text", "columns": ["brief_summary"]}
    (bench.root / "channels.yaml").write_text(yaml.safe_dump(reg))
    fx = freeze(bench.spec_path, out_dir=bench.out)
    with pytest.raises(UnregisteredColumn) as err:
        write_manifest(bench.root / "channels.yaml", fx, bench.root / "channel_manifest.json")
    text = "\n".join(err.value.problems)
    assert "'brief_summary/textblock' is not registered" in text
    assert "registers 'brief_summary', which the table lacks" in text


# 2 ---------------------------------------------------------------- section 10, T1


def test_2_stale_floor_table_is_refused_downstream(bench):
    """Campaign: the noise-floor table was computed before the featurizer repair and never
    recomputed; the re-run overwrote the records behind it and fifteen scripts kept
    reading it (T1; section 10).
    Guard: the table is stamped over the runs, so every reader refuses it after a re-run,
    and so does the verdict step."""
    _small(bench)
    run_grid(bench.grid)
    resolve(bench.grid, n_boot=50)
    questions = {
        "grid": "grid.yaml",
        "resolution": "resolution.json",
        "questions": [
            {
                "id": "Q1",
                "question": "?",
                "rule": "difference_clears_floor",
                "params": {"cell": "mortality/Phase1", "a": "logreg", "b": "logreg"},
            }
        ],
    }
    (bench.root / "questions.yaml").write_text(yaml.safe_dump(questions))

    _touch_code(bench)
    run_grid(bench.grid, on_stale="rerun")  # the re-run after the repair
    with pytest.raises(StaleInput):
        read_resolution(bench.root / "resolution.json")
    with pytest.raises(StaleInput):
        decide(bench.root / "questions.yaml")


# 3 ---------------------------------------------------------------- section 9


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_3_uncommitted_fix_changes_the_code_state(bench):
    """Campaign: analyses ran after the repair but recorded the pre-repair commit, because
    the fix was not yet committed (section 9).
    Guard: the code state is a hash of the working tree. The commit hash stays the same,
    the dirty flag shows, and the old records are refused instead of resumed."""
    _small(bench)

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=bench.root,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "before the repair")
    run_grid(bench.grid)
    before = read_artifact(next(bench.runs.rglob("*.json"))).code

    _touch_code(bench)  # the fix, not committed
    with pytest.raises(GridError, match="stale"):
        run_grid(bench.grid)
    run_grid(bench.grid, on_stale="rerun")
    after = read_artifact(next(bench.runs.rglob("*.json"))).code
    assert after.git_sha == before.git_sha  # the commit hash cannot see the fix
    assert before.git_dirty is False and after.git_dirty is True
    assert after.digest != before.digest


# 4 ---------------------------------------------------------------- sections 3 and 9, T4


def test_4_partial_grid_cannot_be_summarized(bench):
    """Campaign: the leaderboard averaged each method over whichever cells it had finished;
    FT-Transformer ranked second on its finished cells and sixth once the rest ran (T4).
    Guard: coverage counts every planned run, and the resolution step refuses a grid
    with a missing or failed run."""
    _small(bench, methods=["logreg", "hist_gbm"])
    run_grid(bench.grid)
    next(bench.runs.rglob("*Phase3__hist_gbm*.json")).unlink()  # an unfinished cell
    cov = coverage(bench.grid)
    assert cov["planned"] == 6 and cov["counts"]["missing"] == 1 and not cov["complete"]
    with pytest.raises(IncompleteGrid, match="average each method over different cells"):
        resolve(bench.grid, n_boot=50)


# 5 ---------------------------------------------------------------- section 11


def test_5_pooled_bootstrap_must_resample_trials(bench):
    """Campaign: a pooled bootstrap resampled rows, so each trial counted once per seed,
    and intervals were about 2.17 times too narrow (section 11).
    Guard: resampling is clustered on the unit that repeats. Pooling five seeds of real
    grid predictions, the row bootstrap is too narrow and the trial bootstrap is not."""
    _small(bench, methods=["hist_gbm"], fit_seeds=[0, 1, 2, 3, 4])
    run_grid(bench.grid)
    frames = [
        pd.read_csv(p, dtype={"id": str, "cluster": str})
        for p in sorted(bench.runs.rglob("mortality--Phase1__hist_gbm__*.predictions.csv"))
    ]
    pooled = pd.concat(frames)
    pooled = pooled[pooled["split"] == "test"]
    single = frames[0][frames[0]["split"] == "test"]
    from accbench.metrics import roc_auc

    y, s = pooled["label"].to_numpy(), pooled["score"].to_numpy()
    by_row = bootstrap_metric(roc_auc, y, s, np.arange(len(pooled)), n_boot=300)
    by_trial = bootstrap_metric(roc_auc, y, s, pooled["cluster"].to_numpy(), n_boot=300)
    one_seed = bootstrap_metric(
        roc_auc,
        single["label"].to_numpy(),
        single["score"].to_numpy(),
        single["cluster"].to_numpy(),
        n_boot=300,
    )
    assert by_trial.half_width > 1.5 * by_row.half_width
    assert by_trial.half_width == pytest.approx(one_seed.half_width, rel=0.35)


# 6 ---------------------------------------------------------------- section 11


def test_6_selection_on_the_test_set_is_biased_and_labelled():
    """Campaign: a selection step picked the best model on the test set and then
    bootstrapped on the same rows, which biases the result upward (section 11).
    Guard: select() sees validation scores only; oracle_select() is labelled, and a
    confirmatory claim cannot rest on it (test_claims covers the refusal)."""
    rng = np.random.default_rng(0)
    oracle_gain, validation_gain = [], []
    for _ in range(30):
        n, methods = 200, [f"m{i}" for i in range(10)]
        y = rng.integers(0, 2, n)
        rows, val = {}, {}
        for m in methods:  # ten methods of identical skill
            rows[("c", m)] = (y, y * 0.6 + rng.normal(0, 1, n), np.arange(n).astype(str), None)
            val[("c", m)] = float(rng.normal(0.6, 0.02))
        ctx = SyntheticContext(rows, {"c": 0.05}, val)
        mean = np.mean([ctx.goodness(y, rows[("c", m)][1]) for m in methods])
        oracle_gain.append(ctx.goodness(y, rows[("c", ctx.oracle_select("c", methods))][1]) - mean)
        validation_gain.append(ctx.goodness(y, rows[("c", ctx.select("c", methods))][1]) - mean)
        assert ctx.tracking.oracle
    assert np.mean(oracle_gain) > 0.02  # picking on test flatters the winner
    assert abs(np.mean(validation_gain)) < 0.01  # picking on validation does not


# 7 ---------------------------------------------------------------- section 11, T28b


def _gallery_null(seed):
    return _null_drops(seed)


@rule("gallery_separate_drops", null_input=_gallery_null)
def _separate_drops(ctx, *, cell, model, reference, seen_arm, unseen_arm, n_boot=200, seed=0):
    def significant_drop(method):
        seen = bootstrap_metric(
            ctx.goodness, *ctx.test_rows(cell, method, seen_arm), n_boot=n_boot, seed=seed
        )
        unseen = bootstrap_metric(
            ctx.goodness, *ctx.test_rows(cell, method, unseen_arm), n_boot=n_boot, seed=seed
        )
        return unseen.hi < seen.lo

    if significant_drop(model) and not significant_drop(reference):
        return Verdict("effect", "outcome recall demonstrated")
    return Verdict("inconclusive", "no recall shown")


def test_7_each_drop_tested_alone_fails_the_null_input_test():
    """Campaign: the memorization test asked whether the model's drop and the reference's
    drop were each significant on their own, and declared recall where the correct
    verdict was inconclusive (T28b; section 11).
    Guard: the null-input suite. On data where both drop by the same amount, the
    separate-drops rule claims an effect and is refused; difference_of_drops passes."""
    results = run_null_suite(["gallery_separate_drops", "difference_of_drops"])
    assert not results["gallery_separate_drops"]["passed"]
    assert results["difference_of_drops"]["passed"]


# 8 ---------------------------------------------------------------- section 12


def test_8_claim_written_after_its_result_is_refused(bench):
    """Campaign: Part 1 published its findings with no registered decision rules; checked
    later, two of twelve held as published (section 12).
    Guard: the claims register refuses to resolve a claim registered after the first
    run behind its verdict."""
    _small(bench, methods=["logreg", "hist_gbm"])
    questions = {
        "grid": "grid.yaml",
        "resolution": "resolution.json",
        "questions": [
            {
                "id": "Q1",
                "question": "Does logreg beat hist_gbm on Phase 1?",
                "rule": "difference_clears_floor",
                "params": {
                    "cell": "mortality/Phase1",
                    "a": "logreg",
                    "b": "hist_gbm",
                    "n_boot": 100,
                },
            }
        ],
    }
    (bench.root / "questions.yaml").write_text(yaml.safe_dump(questions))
    run_grid(bench.grid)
    resolve(bench.grid, n_boot=50)
    decide(bench.root / "questions.yaml")
    claims.register(
        bench.root / "claims.jsonl",
        claim_id="C1",
        statement="Logistic regression beats gradient boosting on Phase 1.",
        question="Q1",
        prediction="effect:logreg",
        questions=bench.root / "questions.yaml",
        construct=bench.root / "construct.yaml",
    )
    with pytest.raises(claims.ClaimError, match="after the first run"):
        claims.resolve(bench.root / "claims.jsonl", verdicts=bench.root / "verdicts.json")
