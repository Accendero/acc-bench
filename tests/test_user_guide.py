"""Hold docs/user-guide.md to the ASD-STE100 rules that a script can examine.

Examined here: sentence length (20 words in procedural steps, 25 in descriptive text),
paragraph length (6 sentences), no -ing forms except technical names, no contractions,
no dashes as punctuation, no sentence that starts with And, But or So, and a list of
common words that STE does not approve, with the approved word to use.

Not examined here: the full STE dictionary (approved meanings of each word) and the
voice of each sentence. A person reviews those.
"""

import re
import sys
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "user-guide.md"

PROCEDURAL_MAX = 20
DESCRIPTIVE_MAX = 25
PARAGRAPH_MAX = 6

# Common words that STE does not approve, and what to write instead.
NOT_APPROVED = {
    "ensure": "make sure",
    "verify": "make sure / examine",
    "check": "examine / make sure (the command name in code is not text)",
    "obtain": "find / get the result by ...",
    "get": "find / make / show",
    "gets": "find / make / show",
    "need": "must / necessary",
    "needs": "must / necessary",
    "require": "must / necessary",
    "requires": "must / necessary",
    "required": "necessary",
    "want": "(write the purpose: to ...)",
    "allow": "let",
    "allows": "let",
    "provide": "give",
    "provides": "give",
    "create": "make",
    "creates": "make",
    "generate": "make",
    "perform": "do",
    "utilize": "use",
    "via": "through / with",
    "should": "must / (a command)",
    "would": "(simple tense)",
    "could": "can",
    "might": "can",
    "may": "can",
    "please": "(a command)",
    "whether": "if",
    "since": "because / after",
    "once": "after / when",
    "unless": "if ... not",
    "however": "(two sentences)",
    "therefore": "thus / (two sentences)",
    "simply": "(delete)",
    "just": "(delete)",
    "easily": "(delete)",
    "support": "(name the result)",
    "supports": "(name the result)",
    "extends": "is a subclass of",
    "returns": "gives",
}
ING_ALLOWED = {"training", "during", "string", "nothing", "something", "thing", "warning"}
STARTS_FORBIDDEN = ("And ", "But ", "So ")


def _blocks():
    """Yield (kind, text) for each prose unit: 'step', 'text' or 'cell'."""
    in_code = False
    kind, lines = "text", []

    def flush():
        nonlocal kind, lines
        unit = (kind, " ".join(lines)) if lines else None
        kind, lines = "text", []
        return unit

    for raw in GUIDE.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not stripped or stripped.startswith("#"):
            if unit := flush():
                yield unit
            continue
        if stripped.startswith("|"):
            if unit := flush():
                yield unit
            if not set(stripped) <= set("|-: "):
                for cell in stripped.strip("|").split("|"):
                    yield "cell", cell.strip()
            continue
        if re.match(r"\d+\.\s", stripped):
            if unit := flush():
                yield unit
            kind, lines = "step", [re.sub(r"^\d+\.\s", "", stripped)]
            continue
        if kind == "step" and not raw.startswith("   "):
            if unit := flush():
                yield unit
        lines.append(stripped)
    if unit := flush():
        yield unit


def _plain(text: str) -> str:
    text = re.sub(r"`[^`]*`", "CODE", text)  # a code span counts as one word
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links: keep the label
    text = text.replace("**", "")
    return text


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!:])\s+(?=[A-Z0-9\"]|acc-bench\b)", _plain(text))
    return [p.strip() for p in parts if p.strip()]


def _words(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", sentence)


def _problems() -> list[str]:
    problems = []
    for kind, text in _blocks():
        sentences = _sentences(text)
        if kind == "text" and len(sentences) > PARAGRAPH_MAX:
            problems.append(f"paragraph has {len(sentences)} sentences: {text[:60]}...")
        limit = PROCEDURAL_MAX if kind == "step" else DESCRIPTIVE_MAX
        for s in sentences:
            words = _words(s)
            if len(words) > limit:
                problems.append(f"{len(words)} words (max {limit}, {kind}): {s}")
            if s.startswith(STARTS_FORBIDDEN):
                problems.append(f"starts with And/But/So: {s}")
            for w in words:
                lw = w.lower().strip(".'")
                if lw in NOT_APPROVED:
                    problems.append(f"'{w}' is not approved, use {NOT_APPROVED[lw]}: {s}")
                if lw.endswith("ing") and len(lw) > 4 and lw not in ING_ALLOWED:
                    problems.append(f"-ing form '{w}': {s}")
                if re.search(r"n't$|'re$|'ve$|'ll$|'d$", lw):
                    problems.append(f"contraction '{w}': {s}")
        if re.search(r"[—–]|\s-\s", _plain(text)):
            problems.append(f"dash used as punctuation: {text[:60]}...")
    return problems


def test_user_guide_follows_the_ste_rules_a_script_can_examine():
    problems = _problems()
    assert not problems, "\n".join(problems)


def test_the_guide_names_every_command():
    from accbench.cli import build_parser

    text = GUIDE.read_text(encoding="utf-8")
    for command in build_parser()._subparsers._group_actions[0].choices:
        assert f"acc-bench {command}" in text, command


def test_the_checker_finds_violations(tmp_path, monkeypatch):
    bad = tmp_path / "bad.md"
    bad.write_text(
        "# Bad\n\n"
        "1. You should ensure that the file is there before running the command again now.\n"
        "2. " + " ".join(["word"] * 21) + ".\n\n"
        "But this sentence starts badly — and it doesn't stop.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "GUIDE", bad)
    found = "\n".join(_problems())
    for expected in (
        "'should'",
        "'ensure'",
        "-ing form 'running'",
        "21 words (max 20",
        "starts with And/But/So",
        "contraction",
        "dash",
    ):
        assert expected in found, expected


# ---------------------------------------------------------------- the guide's examples run


def _example_files() -> dict[str, str]:
    """The files the guide shows after a line 'Example `<name>`:'."""
    files, want, block, in_code = {}, None, [], False
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"Example `([^`]+)`:\s*$", line.strip())
        if m:
            want = m.group(1)
            continue
        if line.strip().startswith("```"):
            if in_code and want:
                files[want] = "\n".join(block) + "\n"
                want = None
            in_code, block = not in_code, []
            continue
        if in_code:
            block.append(line)
    return files


def _guide_commands(prefix: str) -> list[list[str]]:
    import shlex

    out = []
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            out.append(shlex.split(line.strip())[1:])
    return out


def _patients(path: Path) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    frames = []
    for hospital, shift in (("north", 0.0), ("south", 0.6)):
        n = 700
        age = rng.normal(62, 14, n).round(0)
        stay = rng.integers(1, 20, n)
        ward = rng.choice(["cardiology", "surgery", "general"], n)
        oxygen = rng.random(n) < 0.25
        notes = [
            ("home oxygen " if o else "home ") + rng.choice(["stable", "improving"]) for o in oxygen
        ]
        logit = -1.5 + 0.03 * (age - 62) + 0.08 * stay + shift * (ward == "surgery")
        logit = logit + 1.8 * oxygen + rng.normal(0, 0.5, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": [f"{hospital[0].upper()}{i:05d}" for i in range(n)],
                    "hospital": hospital,
                    "age": age,
                    "length_of_stay": stay,
                    "ward": ward,
                    "discharge_note": notes,
                    "readmitted": y,
                }
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(path, index=False)


def test_the_guide_examples_run_end_to_end(tmp_path, monkeypatch, capsys):
    """Section 6, step by step, with the guide's own example files and commands."""
    from accbench import claims
    from accbench.cli import main

    files = _example_files()
    assert set(files) == {
        "construct.yaml",
        "fixtures.yaml",
        "channels.yaml",
        "grid.yaml",
        "questions.yaml",
    }
    project = tmp_path / "project"
    _patients(project / "data" / "patients.csv")
    for name, text in files.items():
        (project / name).write_text(text, encoding="utf-8")
    (project / "report.md").write_text(
        "The text model is better in the south hospital [claim:C1].\n"
        "It is not better in both hospitals [claim:C2].\n"
        "Age was not tested [claim:C3].\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    def ok(args):
        code = main(args)
        assert code == 0, (args, capsys.readouterr())

    ok(["construct", "check"])
    ok(["fixtures", "freeze"])
    assert "readmitted/south" in capsys.readouterr().out
    ok(["channels", "check"])
    ok(["rules", "test", "questions.yaml"])
    out = capsys.readouterr().out
    assert "match the grid" in out and "FAIL" not in out
    registers = _guide_commands("acc-bench claims register")
    assert [r[2] for r in registers] == ["C1", "C2", "C3"]
    for args in registers:
        ok(args)
    for args in _guide_commands("acc-bench claims untested"):
        ok(args)
    ok(["run"])
    ok(["resolve"])
    ok(["verdict"])
    assert "null-input test passed" in capsys.readouterr().out
    ok(["claims", "resolve"])
    ok(["claims", "show"])
    ok(["claims", "check-doc", "report.md", "--require-all"])
    rows = claims.table(project / "claims.jsonl")
    assert all(r["status"] != "registered" for r in rows)
