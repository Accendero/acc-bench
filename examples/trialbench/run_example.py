"""Run the TrialBench mortality example end to end.

    python examples/trialbench/run_example.py                      # downloads from Zenodo
    python examples/trialbench/run_example.py --from-dir PATH      # an extracted folder
    python examples/trialbench/run_example.py --from-zip PATH      # a downloaded zip

Works in examples/trialbench/out/ (git-ignored). The two claims are registered before the
grid runs, from the campaign's published results, and come back however they come back.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

from accbench.cli import main as acc_bench

HERE = Path(__file__).resolve().parent
INPUTS = [
    "construct.yaml",
    "fixtures.yaml",
    "channels.yaml",
    "grid.yaml",
    "questions.yaml",
    "report.md",
    "download.py",
    "prepare.py",
]
CLAIMS = [
    (
        "TB1",
        "Q1",
        "effect:tfidf_logreg",
        "The text model beats the best tabular model on mortality in Phase 1.",
    ),
    (
        "TB2",
        "Q2",
        "no_effect",
        "The text model does not beat the best tabular model in 3 or more of the 4 phases.",
    ),
]


def step(*args: str) -> None:
    print(f"\n$ acc-bench {shlex.join(args)}", flush=True)
    code = acc_bench(list(args))
    if code != 0:
        raise SystemExit(f"acc-bench {shlex.join(args)} exited with {code}")


def run(workdir: Path, *, from_dir: Path | None = None, from_zip: Path | None = None) -> int:
    workdir.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copy2(HERE / name, workdir / name)
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    import download
    import prepare

    download.fetch(Path("data/raw"), from_dir=from_dir, from_zip=from_zip)
    if not Path("data/mortality.csv").exists():
        prepare.prepare()

    step("construct", "check")
    step("fixtures", "freeze")
    step("channels", "check")
    if not Path("claims.jsonl").exists():  # registered once, before the first run
        for claim_id, question, prediction, statement in CLAIMS:
            step(
                "claims",
                "register",
                claim_id,
                "--statement",
                statement,
                "--question",
                question,
                "--prediction",
                prediction,
            )
    step("run")
    step("resolve")
    step("verdict")
    step("claims", "resolve")
    step("claims", "show")
    step("claims", "check-doc", "report.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-dir", type=Path)
    group.add_argument("--from-zip", type=Path)
    parser.add_argument("--workdir", type=Path, default=HERE / "out")
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.workdir.resolve(),
            from_dir=args.from_dir.resolve() if args.from_dir else None,
            from_zip=args.from_zip.resolve() if args.from_zip else None,
        )
    )
