import copy

import pytest
import yaml

from accbench.cli import main
from accbench.construct import load_construct, require_construct, validate
from accbench.errors import ConstructError

VALID = {
    "version": 1,
    "stated_on": "2026-10-07",
    "owner": "A. Person",
    "question": "Which method should rank our trials for mortality review?",
    "decision": "Which trials a reviewer reads first.",
    "target": {"name": "mortality", "definition": "Any death reported in the trial's results."},
    "metric": {
        "name": "pr_auc",
        "reason": "Only the top of the ranked list is acted on.",
        "alternatives": [{"name": "roc_auc", "why_not": "Scores the whole ranking."}],
    },
    "tasks": [{"name": "mortality", "reason": "The decision concerns mortality only."}],
    "reported_numbers": [
        {"name": "pr_auc_by_cell", "decision": "Which method to deploy for each phase."}
    ],
    "measures": [
        {
            "quantity": "feature importance",
            "used": "drop_and_refit",
            "instead_of": "permutation",
            "reason": "The decision is whether to collect the field.",
        }
    ],
}


def _write(tmp_path, data, name="construct.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def _problems_for(mutate):
    data = copy.deepcopy(VALID)
    mutate(data)
    return validate(data)


def test_valid_statement_passes(tmp_path):
    c = load_construct(_write(tmp_path, VALID))
    assert c.version == 1
    assert c.metric == "pr_auc"
    assert c.tasks == ("mortality",)
    assert len(c.sha256) == 64


def test_yaml_dates_are_accepted(tmp_path):
    p = tmp_path / "construct.yaml"
    text = yaml.safe_dump(VALID, sort_keys=False).replace("'2026-10-07'", "2026-10-07")
    p.write_text(text)
    assert load_construct(p).version == 1


def test_missing_file_blocks_the_freeze(tmp_path):
    with pytest.raises(ConstructError, match="state the construct before freezing"):
        require_construct(tmp_path / "construct.yaml")


def test_invalid_yaml_is_refused(tmp_path):
    p = tmp_path / "construct.yaml"
    p.write_text("version: [1\n")
    with pytest.raises(ConstructError, match="not valid YAML"):
        load_construct(p)


def test_every_problem_is_reported_at_once(tmp_path):
    data = copy.deepcopy(VALID)
    del data["decision"]
    data["metric"]["name"] = "auroc"
    with pytest.raises(ConstructError) as err:
        load_construct(_write(tmp_path, data))
    assert len(err.value.problems) == 2


@pytest.mark.parametrize(
    "key",
    [
        "version",
        "stated_on",
        "owner",
        "question",
        "decision",
        "target",
        "metric",
        "tasks",
        "reported_numbers",
    ],
)
def test_each_required_key_is_required(key):
    assert f"{key}: missing" in _problems_for(lambda d: d.pop(key))


def test_unknown_key_is_an_error():
    # A misspelt key would otherwise be silently ignored.
    assert "reported_number: unknown key" in _problems_for(
        lambda d: d.__setitem__("reported_number", [])
    )


def test_unknown_metric_is_refused():
    problems = _problems_for(lambda d: d["metric"].__setitem__("name", "auroc"))
    assert any("metric.name" in p for p in problems)


def test_metric_needs_a_reason():
    assert "metric.reason: must be a non-empty string" in _problems_for(
        lambda d: d["metric"].__setitem__("reason", " ")
    )


def test_every_reported_number_names_its_decision():
    # R0: every reported number stands in for a decision the reader can name.
    problems = _problems_for(lambda d: d["reported_numbers"][0].pop("decision"))
    assert "reported_numbers[0].decision: missing" in problems


def test_at_least_one_reported_number_and_task():
    assert any(
        "reported_numbers" in p
        for p in _problems_for(lambda d: d.__setitem__("reported_numbers", []))
    )
    assert any("tasks" in p for p in _problems_for(lambda d: d.__setitem__("tasks", [])))


def test_duplicate_task_is_refused():
    problems = _problems_for(lambda d: d["tasks"].append(dict(d["tasks"][0])))
    assert "tasks: 'mortality' is listed more than once" in problems


def test_measure_must_name_what_it_replaces():
    problems = _problems_for(lambda d: d["measures"][0].pop("instead_of"))
    assert "measures[0].instead_of: missing" in problems


def test_bad_version_and_date():
    assert any("version" in p for p in _problems_for(lambda d: d.__setitem__("version", 0)))
    assert any("version" in p for p in _problems_for(lambda d: d.__setitem__("version", True)))
    assert any(
        "stated_on" in p for p in _problems_for(lambda d: d.__setitem__("stated_on", "last week"))
    )


def test_amended_version_needs_its_amendments():
    # An amendment is made in the open: version 3 needs dated entries for 2 and 3.
    def bump(d):
        d["version"] = 3
        d["amendments"] = [{"version": 2, "date": "2026-11-01", "reason": "Added a task."}]

    assert any("amendments" in p for p in _problems_for(bump))

    def full(d):
        bump(d)
        d["amendments"].append({"version": 3, "date": "2026-11-09", "reason": "New cutoff."})

    assert _problems_for(full) == []


def test_amendment_without_bump_is_refused():
    problems = _problems_for(
        lambda d: d.__setitem__("amendments", [{"version": 2, "date": "2026-11-01", "reason": "x"}])
    )
    assert any("amendments" in p for p in problems)


def test_non_mapping_statement():
    assert validate(["not", "a", "mapping"]) == ["the statement must be a YAML mapping"]


def test_cli_check_ok(tmp_path, capsys):
    p = _write(tmp_path, VALID)
    assert main(["construct", "check", str(p)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_check_fails_with_problems(tmp_path, capsys):
    data = copy.deepcopy(VALID)
    del data["owner"]
    p = _write(tmp_path, data)
    assert main(["construct", "check", str(p)]) == 1
    assert "owner: missing" in capsys.readouterr().err
