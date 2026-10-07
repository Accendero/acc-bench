import accbench
from accbench.cli import main


def test_version_is_set():
    assert accbench.__version__


def test_seven_units_in_flow_order():
    assert accbench.UNITS == (
        "construct",
        "fixtures",
        "channels",
        "runner",
        "resolution",
        "rules",
        "claims",
    )


def test_cli_without_unit_prints_help(capsys):
    assert main([]) == 0
    assert "acc-bench" in capsys.readouterr().out


def test_every_unit_has_a_command():
    from accbench.cli import build_parser

    commands = set(build_parser()._subparsers._group_actions[0].choices)
    expected = {
        "construct",
        "fixtures",
        "channels",
        "run",
        "coverage",
        "resolve",
        "rules",
        "verdict",
        "claims",
    }
    assert commands == expected
