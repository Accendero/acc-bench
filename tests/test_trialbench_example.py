"""The TrialBench example's download and prepare steps, on a tiny fake folder.

The real data is never downloaded in tests."""

import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "trialbench"
sys.path.insert(0, str(EXAMPLE))

import download  # noqa: E402
import prepare  # noqa: E402


def _fake(root: Path) -> Path:
    folder = root / "mortality-event-prediction"
    n = 0
    for phase in download.PHASES:
        for split in ("train", "test"):
            d = folder / phase
            d.mkdir(parents=True, exist_ok=True)
            ids = [f"NCT{n + i:08d}" for i in range(4)]
            n += 4
            x = pd.DataFrame({"enrollment": [10, 20, 30, 40], "phase": phase}, index=ids)
            y = pd.DataFrame({"mortality_rate": [0, 0.1, 0, 0.2], "Y/N": [0, 1, 0, 1]}, index=ids)
            x.to_csv(d / f"{split}_x.csv")
            y.to_csv(d / f"{split}_y.csv")
    return folder


def test_from_dir_and_prepare(tmp_path):
    src = _fake(tmp_path / "src")
    raw = tmp_path / "work" / "data" / "raw"
    source = download.fetch(raw, from_dir=src)
    assert source["licence"] == "CC BY 4.0"
    out = prepare.prepare(raw, tmp_path / "work" / "data" / "mortality.csv")
    table = pd.read_csv(out)
    assert len(table) == 32 and table["trial_id"].is_unique
    assert set(table["shipped_split"]) == {"train", "test"}
    assert list(table.columns[:4]) == ["trial_id", "phase_cell", "shipped_split", "label"]
    # A second fetch finds the data in place and does not copy again.
    assert download.fetch(raw)["route"] == "local folder"


def test_zip_with_the_wrong_checksum_is_refused(tmp_path):
    src = _fake(tmp_path / "src")
    zpath = tmp_path / "mortality-event-prediction.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in src.rglob("*.csv"):
            zf.write(f, f.relative_to(src.parent))
    with pytest.raises(SystemExit, match="MD5"):
        download.fetch(tmp_path / "raw", from_zip=zpath)
    assert not (tmp_path / "raw" / "mortality-event-prediction").exists()


def test_wrong_folder_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="missing"):
        download.fetch(tmp_path / "raw", from_dir=tmp_path / "empty")
