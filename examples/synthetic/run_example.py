"""Run the synthetic example end to end and list the seven artifacts it produces.

    python examples/synthetic/run_example.py            # works in examples/synthetic/out
    python examples/synthetic/run_example.py --workdir DIR

The inputs are copied into the work directory first, so the example's own folder holds
only inputs. Claims are registered before the grid runs, as the claims unit requires.
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
    "prices.yaml",
    "questions.yaml",
    "report.md",
    "methods.py",
    "make_data.py",
]
ARTIFACTS = [
    ("0 construct", "construct.yaml"),
    ("1 fixtures", "fixtures/"),
    ("2 channels", "channel_manifest.json"),
    ("3 runner", "runs/ and runs.coverage.json"),
    ("4 resolution", "resolution.json"),
    ("5 rules", "verdicts.json and null_suite.json"),
    ("6 claims", "claims.jsonl"),
]

CLAIMS = [
    ("C1", "Q1", "effect:tfidf_logreg", "The text model beats the best tabular model on Phase 1."),
    ("C2", "Q2", "no_effect", "The text model does not beat the best tabular model across phases."),
    ("C3", "Q1", "effect", "The stand-in language model recalls trial outcomes."),
]


def step(*args: str) -> None:
    print(f"\n$ acc-bench {shlex.join(args)}")
    code = acc_bench(list(args))
    if code != 0:
        raise SystemExit(f"acc-bench {' '.join(args)} exited with {code}")


def run(workdir: Path) -> int:
    """Copy the inputs into ``workdir`` (emptied first) and run every step there."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for name in INPUTS:
        shutil.copy2(HERE / name, workdir / name)
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    import make_data

    make_data.main()

    step("construct", "check")
    step("fixtures", "freeze")
    step("channels", "check")
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
    step(
        "claims",
        "untested",
        "C3",
        "--reason",
        "Recall needs a model with a training cutoff; the stand-in has none.",
    )
    step("run")
    step("resolve", "--n-boot", "300")
    step("verdict")
    step("claims", "resolve")
    step("claims", "show")
    step("claims", "check-doc", "report.md")

    print("\nThe seven artifacts:")
    missing = []
    for unit, paths in ARTIFACTS:
        for p in paths.replace(" and ", ",").split(","):
            if not Path(p.strip()).exists():
                missing.append(p.strip())
        print(f"  {unit:13} {paths}")
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=str(HERE / "out"))
    raise SystemExit(run(Path(parser.parse_args().workdir).resolve()))
