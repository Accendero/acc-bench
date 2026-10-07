# acc-bench

A benchmarking system with seven checked layers between a model and a claim.

acc-bench is the reference implementation of the system in Accendero's whitepaper
*Seven Layers Between a Model and a Claim*. The paper describes a benchmark you run on
your own data as seven components, each with a requirement, a guard that enforces it and
an artifact a reader can ask to see. This package is that system as code. Each guard
refuses the failure it exists to catch, and each failure is one the paper's campaign
actually hit.

## Install

```bash
python -m pip install "git+https://github.com/Accendero/acc-bench"
```

Python 3.10 or later. Core dependencies: numpy, pandas, scikit-learn, scipy, PyYAML.
No deep-learning framework and no cloud SDK.

## See it work

```bash
python examples/synthetic/run_example.py
```

The synthetic example runs every unit on 1,800 trials with planted effects in about 20
seconds and writes all seven artifacts. The [TrialBench example](examples/trialbench/)
runs the same system on real clinical-trial data, which it downloads; none ships here.

## The seven units

| Unit | Paper | Requirement | Guard | Artifact | Command |
|---|---|---|---|---|---|
| 0 Construct | §6 | R0 meaning: every number stands in for a named decision | Schema check; fixtures refuse to freeze without it | `construct.yaml` | `acc-bench construct check` |
| 1 Fixtures | §7 | R1 sameness: same splits, label rule and timing for every method | Keyed splits with a fixed test set; one label rule; declared cutoffs and arms | `fixtures/<hash>/` | `acc-bench fixtures freeze` |
| 2 Channels | §8 | R2 isolation: nothing reaches a model except through a registered path | An unregistered or missing column raises; fitting refuses non-training rows | `channel_manifest.json`, `repairs.jsonl` | `acc-bench channels check` |
| 3 Runner | §9 | R3 completeness: every planned run leaves a record or a reason | Atomic, resumable records with code state, threads and cost; skips recorded | `runs/`, `runs.coverage.json` | `acc-bench run`, `acc-bench coverage` |
| 4 Resolution | §10 | R4 resolution: every difference is read against what the data can resolve | Floors by source (split, fit, test set); clustered bootstrap; refuses stale or partial runs | `resolution.json` | `acc-bench resolve` |
| 5 Rules | §11 | R5 sound conclusions: rules fixed in advance and tested on known answers | Every rule passes a null-input test before any verdict; selection on validation only | `verdicts.json`, `null_suite.json` | `acc-bench verdict`, `acc-bench rules test` |
| 6 Claims | §12 | R6 accountable claims: registered before results, amended in the open | Append-only register; refuses late registration, changed questions and oracle verdicts | `claims.jsonl` | `acc-bench claims ...` |

Provenance (X1) runs through units 3 to 6. Every artifact records the code state it was
written under and the hashes of its inputs. A step that reads an artifact refuses one
whose inputs changed or whose code differs from the code running now.

## How the guards work

- **Code state is the working tree.** A sha256 over the source files on disk, so an
  uncommitted fix changes it. The git commit and a dirty flag are recorded beside it for
  reference only.
- **Nothing has a default branch.** Every column is registered to one channel or ignored
  with a reason. Categorical encoding has no default. A misspelt column name raises.
- **The test set stays fixed.** Training and validation are redrawn under named split
  keys, so the noise floor can be split into its sources.
- **Unmeasured is not zero.** A noise term the grid cannot measure is recorded as
  missing.
- **Rules are tested on data with no effect.** A rule that claims an effect on more than
  3 of 20 no-effect draws, or on its exact null, blocks every verdict. The gate reruns
  when a rule changes.
- **Claims come first.** A claim registered after the first run behind its verdict
  cannot be resolved. An amendment needs a dated construct amendment first.

## The defect gallery

[`docs/defect-gallery.md`](docs/defect-gallery.md) lists the eight failures from the
paper's campaign, each rebuilt on synthetic data in
[`tests/test_defect_gallery.py`](tests/test_defect_gallery.py) and stopped by its guard.
The same page lists three defects the guards caught while this package was being built.

## Limits

- One machine, one process. The runner does not distribute work.
- Methods are scikit-learn models or your own classes registered with
  `accbench.methods.register`. No language-model provider is built in; a method that
  calls one meters each call on the pinned price table.
- Binary and multiclass classification only.
- Amending a claim requires amending the construct statement, which in turn requires
  freezing the fixture again before new runs. This is deliberate and strict.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Citing

See [`CITATION.cff`](CITATION.cff).

## Licence

MIT. See [LICENSE](LICENSE). No benchmark data ships with this repository. The
TrialBench example downloads data licensed CC BY 4.0 and says how to cite it.
