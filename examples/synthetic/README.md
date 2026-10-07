# Synthetic example

A complete benchmark on synthetic data with planted effects, so every claim has a known
right answer. No download, about 20 seconds on a laptop.

```bash
python examples/synthetic/run_example.py
```

It copies the inputs into `examples/synthetic/out/` and runs every step there:

| Step | Command | Writes |
|---|---|---|
| Data | `make_data.py` | `data/trials.csv` (1,800 trials, three phases) |
| 0 Construct | `acc-bench construct check` | checks `construct.yaml` |
| 1 Fixtures | `acc-bench fixtures freeze` | `fixtures/<hash>/` |
| 2 Channels | `acc-bench channels check` | `channel_manifest.json` |
| 6 Claims | `acc-bench claims register ...` | `claims.jsonl`, before any run |
| 3 Runner | `acc-bench run` | `runs/`, `runs.coverage.json` |
| 4 Resolution | `acc-bench resolve` | `resolution.json` |
| 5 Rules | `acc-bench verdict` | `null_suite.json`, `verdicts.json` |
| 6 Claims | `acc-bench claims resolve`, `show`, `check-doc report.md` | resolutions in `claims.jsonl` |

What is planted (`make_data.py`): a keyword in the trial summary raises the risk of death
sharply in Phase 1 and barely in Phases 2 and 3; age, enrollment and sponsor carry a
modest signal everywhere. Claim C1 (the text model wins on Phase 1) and claim C2 (it does
not win across phases) should both come back confirmed. C3 is recorded as untested, with
its reason.

What to look at:

- `resolution.json`: every cell's floor is set by the test set (about 150 trials per
  cell), the fit term is zero because these methods ignore the seed, and no cell can
  resolve a difference of 0.05.
- `verdicts.json`: Q1 selects the tabular comparator on validation scores and records
  the choice.
- `runs/mortality/*stub_llm*.json`: the stand-in language model's cost, metered per call
  against the fictional prices in `prices.yaml`.
