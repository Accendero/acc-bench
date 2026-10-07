"""Command-line entry point: ``acc-bench <unit> <verb>``."""

from __future__ import annotations

import argparse
import sys

from accbench import __version__
from accbench.errors import AccbenchError


def _construct_check(args: argparse.Namespace) -> int:
    from accbench.construct import load_construct

    construct = load_construct(args.path)
    print(f"{args.path}: OK ({construct.summary()})")
    return 0


def _fixtures_freeze(args: argparse.Namespace) -> int:
    from accbench.fixtures import freeze

    fixture = freeze(args.spec, out_dir=args.out)
    print(f"{fixture.path}: {fixture.summary()}")
    return 0


def _fixtures_show(args: argparse.Namespace) -> int:
    from accbench.fixtures import load_fixture

    fixture = load_fixture(args.path)
    print(f"{fixture.path}: {fixture.summary()}")
    for task, t in fixture.manifest.data["tasks"].items():
        excluded = ", ".join(f"{n} {r}" for r, n in t["excluded"].items()) or "none"
        print(f"  {task}: label rule {t['label_rule']}; excluded: {excluded}")
    return 0


def _channels_check(args: argparse.Namespace) -> int:
    from pathlib import Path

    from accbench.channels import write_manifest
    from accbench.runner import open_fixture

    fixture = open_fixture(Path(args.fixture), Path(args.fixtures_dir))
    manifest = write_manifest(args.registry, fixture, args.out)
    for task, t in manifest["tasks"].items():
        n = sum(1 for owner in t["columns"].values() if owner != "ignore")
        print(f"{task}: {n} column(s) in {len(t['channels'])} channel(s); every column registered")
    print(f"wrote {args.out}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from accbench.runner import coverage, run_grid

    def progress(key, status):
        print(f"{status:8} {key.cell} {key.method} {key.split_key} seed {key.fit_seed}")

    counts = run_grid(
        args.grid, on_stale="rerun" if args.rerun_stale else "refuse", progress=progress
    )
    print("runs: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    cov = coverage(args.grid)
    print(_coverage_line(cov))
    return 0 if cov["complete"] else 1


def _coverage_line(cov) -> str:
    c = cov["counts"]
    return (
        f"coverage: {cov['planned']} planned = ok {c['ok']} + skipped {c['skipped']} + "
        f"error {c['error']} + missing {c['missing']} + stale {c['stale']}"
        f"{'' if cov['complete'] else '  (INCOMPLETE)'}"
    )


def _coverage(args: argparse.Namespace) -> int:
    from accbench.runner import coverage

    cov = coverage(args.grid)
    print(_coverage_line(cov))
    for method, c in sorted(cov["by_method"].items()):
        print(f"  {method:16} " + "  ".join(f"{k} {v}" for k, v in c.items() if v))
    return 0 if cov["complete"] else 1


def _resolve(args: argparse.Namespace) -> int:
    from accbench.resolution import resolve

    table = resolve(
        args.grid,
        out=args.out,
        methods=args.methods,
        n_boot=args.n_boot,
        threshold=args.threshold,
    )
    print(f"{'cell':28} {'floor':>8}  source  {'split':>7} {'fit':>7} {'test':>7}")

    def fmt(v):
        return "     --" if v is None else f"{v:7.4f}"

    for row in table["cells"]:
        t = row["terms"]
        flag = "" if row["resolves_threshold"] else f"  cannot resolve {table['threshold']}"
        print(
            f"{row['cell']:28} {row['floor']:8.4f}  {row['source']:6} "
            f"{fmt(t['split'])} {fmt(t['fit'])} {fmt(t['test'])}{flag}"
        )
    return 0


def _rules_test(args: argparse.Namespace) -> int:
    from accbench.rules import load_questions, registered_rules, run_null_suite

    if args.questions:
        load_questions(args.questions)  # registers the project's own rules
    results = run_null_suite(args.rule or registered_rules(), draws=args.draws)
    for name, r in results.items():
        verdict = "pass" if r["passed"] else "FAIL"
        exact = (
            ""
            if r["exact_null_claimed_effect"] is None
            else (", exact null: effect" if r["exact_null_claimed_effect"] else ", exact null: ok")
        )
        print(
            f"{verdict}  {name}: effect on {r['effects_on_null']} of {r['draws']} null draws{exact}"
        )
    return 0 if all(r["passed"] for r in results.values()) else 1


def _verdict(args: argparse.Namespace) -> int:
    from accbench.rules import decide

    result = decide(args.questions, out=args.out)
    for v in result["verdicts"]:
        oracle = "  [ORACLE]" if v["oracle"] else ""
        print(f"{v['id']}: {v['outcome']} - {v['summary']}{oracle}")
    return 0


def _claims(args: argparse.Namespace) -> int:
    from accbench import claims

    if args.verb == "register":
        e = claims.register(
            args.log,
            claim_id=args.claim,
            statement=args.statement,
            question=args.question,
            prediction=args.prediction,
            questions=args.questions,
            construct=args.construct,
            exploratory=args.exploratory,
        )
        print(f"registered {e['claim_id']} at {e['recorded_at']}")
    elif args.verb == "amend":
        e = claims.amend(
            args.log,
            claim_id=args.claim,
            reason=args.reason,
            construct=args.construct,
            prediction=args.prediction,
            question=args.question,
            questions=args.questions,
        )
        print(f"amended {e['claim_id']} at {e['recorded_at']}")
    elif args.verb == "untested":
        e = claims.mark_untested(args.log, claim_id=args.claim, reason=args.reason)
        print(f"{e['claim_id']} marked untested")
    elif args.verb == "resolve":
        for e in claims.resolve(args.log, verdicts=args.verdicts):
            print(f"{e['claim_id']}: {e['status']} ({e['question']}: {e['outcome']})")
    elif args.verb == "show":
        rows = claims.table(args.log)
        for r in rows:
            print(f"{r['claim']:10} {r['status']:12} {r['test']:8} {r['statement']}")
        print("  ".join(f"{k} {v}" for k, v in claims.counts(rows).items()))
    elif args.verb == "check-doc":
        cited = claims.check_document(args.log, args.document)
        print(f"{args.document}: {len(cited)} claim(s) cited, all in the register with a result")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acc-bench",
        description="Seven checked layers between a model and a claim.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    units = parser.add_subparsers(dest="unit", metavar="<unit>")

    construct = units.add_parser("construct", help="unit 0: the construct statement")
    verbs = construct.add_subparsers(dest="verb", metavar="<verb>", required=True)
    check = verbs.add_parser("check", help="check a construct statement against its schema")
    check.add_argument("path", nargs="?", default="construct.yaml")
    check.set_defaults(func=_construct_check)

    fixtures = units.add_parser("fixtures", help="unit 1: frozen splits, labels and arms")
    verbs = fixtures.add_subparsers(dest="verb", metavar="<verb>", required=True)
    fr = verbs.add_parser("freeze", help="freeze the fixture a spec describes")
    fr.add_argument("spec", nargs="?", default="fixtures.yaml")
    fr.add_argument("--out", default="fixtures", help="directory for frozen fixtures")
    fr.set_defaults(func=_fixtures_freeze)
    show = verbs.add_parser("show", help="check a frozen fixture and say what was frozen when")
    show.add_argument("path")
    show.set_defaults(func=_fixtures_show)

    channels = units.add_parser("channels", help="unit 2: the channel registry")
    verbs = channels.add_subparsers(dest="verb", metavar="<verb>", required=True)
    ck = verbs.add_parser("check", help="check source tables against the registry")
    ck.add_argument("registry", nargs="?", default="channels.yaml")
    ck.add_argument(
        "--fixture", default="fixtures.yaml", help="the fixture spec, or a frozen fixture directory"
    )
    ck.add_argument("--fixtures-dir", default="fixtures", help="where frozen fixtures live")
    ck.add_argument("--out", default="channel_manifest.json")
    ck.set_defaults(func=_channels_check)

    run = units.add_parser("run", help="unit 3: run the grid (resumable)")
    run.add_argument("grid", nargs="?", default="grid.yaml")
    run.add_argument("--rerun-stale", action="store_true", help="rerun stale records")
    run.set_defaults(func=_run)
    cov = units.add_parser("coverage", help="unit 3: count planned runs by status")
    cov.add_argument("grid", nargs="?", default="grid.yaml")
    cov.set_defaults(func=_coverage)

    res = units.add_parser("resolve", help="unit 4: noise floors by source")
    res.add_argument("grid", nargs="?", default="grid.yaml")
    res.add_argument("--out", default=None, help="default: resolution.json beside the grid")
    res.add_argument("--methods", nargs="+", default=None, help="default: all but majority")
    res.add_argument("--n-boot", type=int, default=1000)
    res.add_argument("--threshold", type=float, default=0.05)
    res.set_defaults(func=_resolve)

    rules = units.add_parser("rules", help="unit 5: decision rules and their null-input tests")
    verbs = rules.add_subparsers(dest="verb", metavar="<verb>", required=True)
    rt = verbs.add_parser("test", help="run the null-input suite")
    rt.add_argument("questions", nargs="?", default=None, help="loads the project's rules module")
    rt.add_argument("--rule", nargs="+", default=None)
    rt.add_argument("--draws", type=int, default=20)
    rt.set_defaults(func=_rules_test)
    vd = units.add_parser("verdict", help="unit 5: answer each pre-registered question")
    vd.add_argument("questions", nargs="?", default="questions.yaml")
    vd.add_argument("--out", default=None, help="default: verdicts.json beside the questions")
    vd.set_defaults(func=_verdict)

    claims = units.add_parser("claims", help="unit 6: the claims register")
    claims.add_argument("--log", default="claims.jsonl")
    verbs = claims.add_subparsers(dest="verb", metavar="<verb>", required=True)
    reg = verbs.add_parser("register", help="register a claim before its result exists")
    reg.add_argument("claim")
    reg.add_argument("--statement", required=True)
    reg.add_argument("--question", required=True, help="the question id that decides it")
    reg.add_argument("--prediction", required=True, help="effect, effect:<method> or no_effect")
    reg.add_argument("--questions", default="questions.yaml")
    reg.add_argument("--construct", default="construct.yaml")
    reg.add_argument("--exploratory", action="store_true")
    am = verbs.add_parser("amend", help="amend a claim in the open, with its reason")
    am.add_argument("claim")
    am.add_argument("--reason", required=True)
    am.add_argument("--prediction")
    am.add_argument("--question")
    am.add_argument("--questions", default="questions.yaml")
    am.add_argument("--construct", default="construct.yaml")
    un = verbs.add_parser("untested", help="record that a claim will not be tested, and why")
    un.add_argument("claim")
    un.add_argument("--reason", required=True)
    rs = verbs.add_parser("resolve", help="resolve claims from a verdicts file")
    rs.add_argument("--verdicts", default="verdicts.json")
    verbs.add_parser("show", help="print the register")
    cd = verbs.add_parser("check-doc", help="check that a document cites only backed claims")
    cd.add_argument("document")
    claims.set_defaults(func=_claims)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.unit is None:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except AccbenchError as exc:
        print(f"acc-bench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
