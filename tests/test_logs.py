import json

import pytest

from accbench.appendlog import append, read_log
from accbench.errors import LogError
from accbench.repairs import read_repairs, record_repair


def test_append_and_read(tmp_path):
    log = tmp_path / "log.jsonl"
    assert read_log(log) == []
    first = append(log, {"kind": "x", "n": 1})
    second = append(log, {"kind": "x", "n": 2})
    entries = read_log(log)
    assert [e["seq"] for e in entries] == [1, 2]
    assert second["prev"] == first["hash"]


@pytest.mark.parametrize("edit", ["change", "delete", "reorder"])
def test_editing_the_log_breaks_the_chain(tmp_path, edit):
    log = tmp_path / "log.jsonl"
    for n in range(3):
        append(log, {"n": n})
    lines = log.read_text().splitlines()
    if edit == "change":
        entry = json.loads(lines[1])
        entry["n"] = 99
        lines[1] = json.dumps(entry, sort_keys=True)
    elif edit == "delete":
        del lines[1]
    else:
        lines[0], lines[1] = lines[1], lines[0]
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(LogError, match="hash chain"):
        read_log(log)
    with pytest.raises(LogError):
        append(log, {"n": 4})


def test_reserved_keys(tmp_path):
    with pytest.raises(LogError, match="may not set"):
        append(tmp_path / "log.jsonl", {"hash": "x"})


def test_repair_needs_before_and_after(tmp_path):
    log = tmp_path / "repairs.jsonl"
    with pytest.raises(LogError, match="before and after"):
        record_repair(
            log, description="fix", columns=["c"], metric="pr_auc", before={}, after={"a": 1}
        )
    with pytest.raises(LogError, match="same cells"):
        record_repair(
            log,
            description="fix",
            columns=["c"],
            metric="pr_auc",
            before={"a": 0.4},
            after={"a": 0.5, "b": 0.5},
        )
    assert read_repairs(log) == []


def test_repair_records_its_measured_effect(tmp_path):
    log = tmp_path / "repairs.jsonl"
    entry = record_repair(
        log,
        description="Register brief_summary/textblock as text; it had fallen into target encoding.",
        columns=["brief_summary/textblock"],
        metric="pr_auc",
        before={"mortality/Phase1": 0.40, "mortality/Phase2": 0.50},
        after={"mortality/Phase1": 0.45, "mortality/Phase2": 0.52},
        evidence=["runs/before", "runs/after"],
    )
    assert entry["delta"] == pytest.approx({"mortality/Phase1": 0.05, "mortality/Phase2": 0.02})
    assert entry["mean_delta"] == pytest.approx(0.035)
    assert read_repairs(log)[0]["kind"] == "repair"
