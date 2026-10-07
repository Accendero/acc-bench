"""Fetch TrialBench's mortality task into data/raw/. No TrialBench data ships with acc-bench.

    python download.py                       # from Zenodo (29.4 MB), checked against its MD5
    python download.py --from-dir PATH       # an extracted mortality-event-prediction folder
    python download.py --from-zip PATH       # a mortality-event-prediction.zip already on disk

Source: Hu, Y. "TrialBench: Benchmarking Multi-Modal Artificial-Intelligence-Ready Clinical
Trial Prediction". Zenodo, 2025. https://doi.org/10.5281/zenodo.15455785. Licence CC BY 4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RECORD = "15455785"
FOLDER = "mortality-event-prediction"
URL = f"https://zenodo.org/records/{RECORD}/files/{FOLDER}.zip?download=1"
SIZE = 29_366_599
MD5 = "f0a9bad80d5abf5b6b8c42d0c7512c4e"
PHASES = ("Phase1", "Phase2", "Phase3", "Phase4")
FILES = ("train_x.csv", "train_y.csv", "test_x.csv", "test_y.csv")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_layout(folder: Path) -> None:
    missing = [f"{p}/{f}" for p in PHASES for f in FILES if not (folder / p / f).is_file()]
    if missing:
        raise SystemExit(f"{folder} is not a {FOLDER} folder; missing {missing[:3]}")


def _extract(zip_path: Path, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
        entries = list(Path(tmp).iterdir())
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(tmp)
        _check_layout(src)
        shutil.copytree(src, dest)


def fetch(raw: Path, *, from_dir: Path | None = None, from_zip: Path | None = None) -> dict:
    dest = raw / FOLDER
    if dest.exists():
        _check_layout(dest)
        print(f"{dest} already present")
        return json.loads((raw / "SOURCE.json").read_text())
    raw.mkdir(parents=True, exist_ok=True)
    if from_dir is not None:
        _check_layout(from_dir)
        shutil.copytree(from_dir, dest)
        source = {"route": "local folder", "path": str(from_dir)}
    else:
        zip_path = from_zip
        tmp_dir = None
        if zip_path is None:
            tmp_dir = Path(tempfile.mkdtemp())
            zip_path = tmp_dir / f"{FOLDER}.zip"
            print(f"downloading {URL} ({SIZE / 1e6:.1f} MB)")
            with urllib.request.urlopen(URL, timeout=300) as resp, zip_path.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        try:
            digest = _md5(zip_path)
            if digest != MD5:
                raise SystemExit(f"{zip_path} has MD5 {digest}, expected {MD5}; not using it")
            _extract(zip_path, dest)
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        source = {"route": "zip" if from_zip else "zenodo", "url": URL, "md5": MD5}
    source.update(
        {
            "record": f"https://doi.org/10.5281/zenodo.{RECORD}",
            "licence": "CC BY 4.0",
            "citation": "Hu, Y. TrialBench: Benchmarking Multi-Modal Artificial-Intelligence-Ready "
            "Clinical Trial Prediction. Zenodo, 2025.",
        }
    )
    (raw / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"wrote {dest}")
    return source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-dir", type=Path)
    group.add_argument("--from-zip", type=Path)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    fetch(args.raw, from_dir=args.from_dir, from_zip=args.from_zip)


if __name__ == "__main__":
    main()
