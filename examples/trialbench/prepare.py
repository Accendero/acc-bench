"""Turn TrialBench's per-phase files into the one source table the fixture reads.

data/raw/mortality-event-prediction/Phase*/{train,test}_{x,y}.csv  ->  data/mortality.csv

Each row keeps every one of TrialBench's 60 feature columns and gains four:
``trial_id`` (the NCT number), ``phase_cell`` (the folder it came from), ``shipped_split``
(TrialBench's own train/test split, which the fixture keeps as the test set) and
``label`` (TrialBench's ``Y/N``). Nothing is dropped, filled or re-encoded here; what
reaches a model is the channel registry's business.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from download import FOLDER, PHASES


def prepare(raw: Path = Path("data/raw"), out: Path = Path("data/mortality.csv")) -> Path:
    frames = []
    for phase in PHASES:
        for split in ("train", "test"):
            folder = raw / FOLDER / phase
            x = pd.read_csv(folder / f"{split}_x.csv", index_col=0, low_memory=False)
            y = pd.read_csv(folder / f"{split}_y.csv", index_col=0)
            if not x.index.equals(y.index):
                raise SystemExit(f"{folder}: {split}_x and {split}_y list different trials")
            frame = x.copy()
            frame.insert(0, "label", y["Y/N"].astype(int))
            frame.insert(0, "shipped_split", split)
            frame.insert(0, "phase_cell", phase)
            frame.insert(0, "trial_id", x.index.astype(str))
            frames.append(frame.reset_index(drop=True))
    table = pd.concat(frames, ignore_index=True)
    if table["trial_id"].duplicated().any():
        raise SystemExit("a trial appears in more than one phase or split")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    counts = table.groupby(["phase_cell", "shipped_split"]).size().unstack()
    print(f"wrote {out}: {len(table)} trials\n{counts}")
    return out


if __name__ == "__main__":
    prepare()
