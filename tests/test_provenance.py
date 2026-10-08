import json
import shutil
import subprocess

import pytest

from accbench.errors import CodeStateMismatch, MissingProvenance, StaleInput
from accbench.provenance import (
    STAMP_KEY,
    code_state,
    file_digest,
    read_artifact,
    require_same_code,
    write_artifact,
)


def _project(tmp_path):
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "featurize.py").write_text("COLUMNS = ['brief_summary']\n")
    (root / "run.py").write_text("print('run')\n")
    return root


def _state(root):
    return code_state([root], include_accbench=False)


# ---------------------------------------------------------------- code state


def test_code_state_is_stable(tmp_path):
    root = _project(tmp_path)
    assert _state(root).digest == _state(root).digest
    assert _state(root).files == 2


def test_uncommitted_edit_changes_code_state(tmp_path):
    # Section 9: a fix left uncommitted is invisible to a commit hash.
    # The working-tree digest sees it.
    root = _project(tmp_path)
    before = _state(root)
    (root / "pkg" / "featurize.py").write_text("COLUMNS = ['brief_summary/textblock']\n")
    assert _state(root).digest != before.digest


def test_line_endings_do_not_change_code_state(tmp_path):
    root = _project(tmp_path)
    f = root / "run.py"
    f.write_bytes(b"import os\nprint('run')\n")
    before = _state(root)
    f.write_bytes(b"import os\r\nprint('run')\r\n")
    assert _state(root).digest == before.digest


def test_non_source_files_and_excluded_dirs_are_ignored(tmp_path):
    root = _project(tmp_path)
    before = _state(root)
    (root / "notes.md").write_text("notes")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "site.py").write_text("x = 1\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "run.py").write_text("x = 1\n")
    assert _state(root).digest == before.digest


def test_renaming_a_file_changes_code_state(tmp_path):
    root = _project(tmp_path)
    before = _state(root)
    (root / "run.py").rename(root / "main.py")
    assert _state(root).digest != before.digest


def test_accbench_source_is_included_by_default(tmp_path):
    root = _project(tmp_path)
    with_lib = code_state([root])
    without = _state(root)
    assert with_lib.files > without.files
    assert with_lib.digest != without.digest


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_sha_and_dirty_flag_are_recorded(tmp_path):
    root = _project(tmp_path)

    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    clean = _state(root)
    assert clean.git_sha and clean.git_dirty is False

    (root / "run.py").write_text("print('fixed')\n")
    dirty = _state(root)
    assert dirty.git_sha == clean.git_sha  # the commit hash cannot see the fix ...
    assert dirty.git_dirty is True  # ... the dirty flag can ...
    assert dirty.digest != clean.digest  # ... and the code state changes.


def test_no_git_is_recorded_as_unknown(tmp_path):
    root = _project(tmp_path)
    state = _state(root)
    if state.git_sha is None:
        assert state.git_dirty is None


# ---------------------------------------------------------------- write and read


def test_round_trip(tmp_path):
    root = _project(tmp_path)
    code = _state(root)
    runs = tmp_path / "runs.json"
    runs.write_text("[1, 2, 3]")
    out = tmp_path / "out" / "resolution.json"
    write_artifact(out, {"floor": 0.05}, code=code, inputs={"runs": runs})

    art = read_artifact(out, expected_code=code)
    assert art.data == {"floor": 0.05}
    assert art.code.digest == code.digest
    assert art.inputs["runs"]["sha256"] == file_digest(runs)
    assert art.overrides == ()


def test_stale_floor_table_is_refused(tmp_path):
    # Section 10 / T1: a noise-floor table outlived the re-run of the records
    # behind it, and fifteen scripts read it. Here the read is refused.
    root = _project(tmp_path)
    code = _state(root)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "cell_a.json").write_text('{"pr_auc": 0.41}')
    table = tmp_path / "resolution.json"
    write_artifact(table, {"delta_cell": 0.09}, code=code, inputs={"runs": runs})

    (runs / "cell_a.json").write_text('{"pr_auc": 0.47}')  # re-run after a repair
    with pytest.raises(StaleInput, match="stale"):
        read_artifact(table)


def test_deleted_input_is_refused(tmp_path):
    code = _state(_project(tmp_path))
    src = tmp_path / "fixture.csv"
    src.write_text("id,y\n1,0\n")
    out = tmp_path / "a.json"
    write_artifact(out, {}, code=code, inputs={"fixture": src})
    src.unlink()
    with pytest.raises(StaleInput, match="no longer exists"):
        read_artifact(out)


def test_missing_input_at_write_is_refused(tmp_path):
    code = _state(_project(tmp_path))
    with pytest.raises(StaleInput):
        write_artifact(tmp_path / "a.json", {}, code=code, inputs={"x": tmp_path / "nope"})


def test_unstamped_artifact_is_refused(tmp_path):
    out = tmp_path / "leaderboard.json"
    out.write_text(json.dumps({"rank": [1, 2]}))
    with pytest.raises(MissingProvenance):
        read_artifact(out)


def test_code_state_mismatch_is_refused(tmp_path):
    root = _project(tmp_path)
    old = _state(root)
    out = tmp_path / "a.json"
    write_artifact(out, {}, code=old)
    (root / "run.py").write_text("print('changed')\n")
    with pytest.raises(CodeStateMismatch, match="the code changed"):
        read_artifact(out, expected_code=_state(root))


def test_code_drift_override_is_recorded_and_carried(tmp_path):
    root = _project(tmp_path)
    old = _state(root)
    first = tmp_path / "a.json"
    write_artifact(first, {}, code=old)
    (root / "run.py").write_text("print('changed')\n")
    new = _state(root)

    art = read_artifact(first, expected_code=new, allow_code_drift=True)
    assert len(art.overrides) == 1 and "code drift" in art.overrides[0]

    second = tmp_path / "b.json"
    write_artifact(second, {}, code=new, inputs={"a": first}, overrides=art.overrides)
    assert json.loads(second.read_text())[STAMP_KEY]["overrides"] == list(art.overrides)
    assert read_artifact(second).overrides == art.overrides


def test_mixed_code_states_are_refused(tmp_path):
    # Records from before and after a repair must not be pooled into one table.
    root = _project(tmp_path)
    a = tmp_path / "a.json"
    write_artifact(a, {}, code=_state(root))
    (root / "pkg" / "featurize.py").write_text("COLUMNS = ['brief_summary/textblock']\n")
    b = tmp_path / "b.json"
    write_artifact(b, {}, code=_state(root))

    arts = [read_artifact(a), read_artifact(b)]
    with pytest.raises(CodeStateMismatch, match="different code states"):
        require_same_code(arts)
    assert require_same_code([arts[1]]).digest == arts[1].code.digest


def test_moved_output_tree_still_checks(tmp_path):
    code = _state(_project(tmp_path))
    out = tmp_path / "out"
    (out / "runs").mkdir(parents=True)
    (out / "runs" / "r.json").write_text("{}")
    write_artifact(out / "resolution.json", {}, code=code, inputs={"runs": out / "runs"})

    moved = tmp_path / "moved"
    shutil.move(str(out), str(moved))
    assert read_artifact(moved / "resolution.json").inputs["runs"]["path"] == "runs"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    code = _state(_project(tmp_path))
    out = tmp_path / "d" / "a.json"
    write_artifact(out, {"x": 1}, code=code)
    write_artifact(out, {"x": 2}, code=code)
    assert [p.name for p in out.parent.iterdir()] == ["a.json"]
    assert read_artifact(out).data == {"x": 2}


def test_payload_with_existing_stamp_is_rejected(tmp_path):
    code = _state(_project(tmp_path))
    with pytest.raises(ValueError):
        write_artifact(tmp_path / "a.json", {STAMP_KEY: {}}, code=code)


def test_directory_digest_covers_names_and_contents(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("1")
    first = file_digest(d)
    (d / "a.txt").rename(d / "b.txt")
    second = file_digest(d)
    (d / "b.txt").write_text("2")
    assert len({first, second, file_digest(d)}) == 3
