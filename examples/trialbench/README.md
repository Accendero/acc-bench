# TrialBench mortality example

The system on real data: TrialBench's mortality task, all four phases, four scikit-learn
methods. About 1.5 minutes on a laptop once the data is in place.

No TrialBench data ships with acc-bench. `download.py` fetches it.

```bash
python examples/trialbench/run_example.py                     # downloads 29.4 MB from Zenodo
python examples/trialbench/run_example.py --from-zip PATH     # or a zip you already have
python examples/trialbench/run_example.py --from-dir PATH     # or an extracted folder
```

The download is checked against Zenodo's published MD5 before it is used. Everything is
written to `examples/trialbench/out/`, which git ignores.

## Data and licence

Hu, Y. *TrialBench: Benchmarking Multi-Modal Artificial-Intelligence-Ready Clinical Trial
Prediction.* Zenodo, 2025. <https://doi.org/10.5281/zenodo.15455785>. Licensed CC BY 4.0.
The labels and fields come from ClinicalTrials.gov.

## What the example does

| Step | Writes |
|---|---|
| `download.py`, `prepare.py` | `data/raw/`, then `data/mortality.csv`: 17,916 trials with TrialBench's 60 fields plus `trial_id`, `phase_cell`, `shipped_split` and `label` |
| `acc-bench fixtures freeze` | TrialBench's own test split kept fixed; validation redrawn under three split keys |
| `acc-bench channels check` | all 64 columns registered: 51 in six channels, 13 ignored with reasons (the city column among them, after the campaign's T20) |
| `acc-bench claims register` | two claims, registered before the first run |
| `acc-bench run` | 48 runs: 4 phases x 4 methods x 3 split keys |
| `acc-bench resolve`, `verdict`, `claims resolve` | floors, verdicts and the claims register |

The two claims were written from the paper's published results before this example was
first run, and they come back however they come back. On the first run, TB1 (the text
model wins on Phase 1) was inconclusive: Phase 1's floor of about 0.07 is wider than the
difference. TB2 (no win in 3 or more phases) was confirmed.
