"""Write a synthetic trials table with known effects to data/trials.csv.

Planted, so the example's claims have a right answer:

- the free-text summary carries most of the signal in Phase 1 (a keyword raises the risk
  of death sharply there) and little in Phases 2 and 3;
- the structured fields (age, enrollment, sponsor) carry a modest signal in every phase.

So a text model should beat the best tabular model on Phase 1, and not across phases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYWORD_EFFECT = {"Phase1": 3.0, "Phase2": 0.3, "Phase3": 0.3}
WORDS = ["oral", "dose", "adults", "cohort", "open", "label", "placebo", "weekly", "arm"]


def make(n_per_phase: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for p, (phase, effect) in enumerate(KEYWORD_EFFECT.items()):
        n = n_per_phase
        age = rng.normal(55, 12, n).round(1)
        enrollment = rng.integers(20, 3000, n)
        sponsor = rng.choice(["industry", "academic", "government"], n, p=[0.6, 0.3, 0.1])
        hepatic = rng.random(n) < 0.3
        summary = [
            " ".join(rng.choice(WORDS, 6)) + (" hepatic impairment" if h else " renal function")
            for h in hepatic
        ]
        logit = (
            -1.6
            + 0.035 * (age - 55)
            + 0.5 * (sponsor == "academic")
            + effect * hepatic
            + rng.normal(0, 0.7, n)
        )
        died = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        registered = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2550, n), "D")
        posted = registered + pd.to_timedelta(rng.integers(300, 1500, n), "D")
        frames.append(
            pd.DataFrame(
                {
                    "trial_id": [f"SYN{p}{i:05d}" for i in range(n)],
                    "phase": phase,
                    "age": age,
                    "enrollment": enrollment,
                    "sponsor": sponsor,
                    "summary": summary,
                    "died": died,
                    "registered_on": registered.strftime("%Y-%m-%d"),
                    "results_posted_on": posted.strftime("%Y-%m-%d"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main(out: str = "data/trials.csv") -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
