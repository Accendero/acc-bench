import os
import sys
from pathlib import Path

from accbench import claims

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic"


def test_synthetic_example_end_to_end(tmp_path):
    """The example runs every unit and its planted claims come back as designed."""
    sys.path.insert(0, str(EXAMPLE))
    cwd = os.getcwd()
    try:
        import run_example

        assert run_example.run(tmp_path / "work") == 0
    finally:
        os.chdir(cwd)
        sys.path.remove(str(EXAMPLE))
    work = tmp_path / "work"
    rows = {r["claim"]: r["status"] for r in claims.table(work / "claims.jsonl")}
    assert rows == {"C1": "confirmed", "C2": "confirmed", "C3": "untested"}
