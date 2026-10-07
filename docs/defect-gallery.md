# Defect gallery

Each failure from the campaign in *Seven Layers Between a Model and a Claim*, rebuilt on synthetic data in [`tests/test_defect_gallery.py`](../tests/test_defect_gallery.py), with the guard that stops it.

| # | Campaign failure | Paper | Guard | Unit |
|---|---|---|---|---|
| 1 | A misnamed text column fell into a default branch, was target-encoded and leaked the label | §8, T7, T8 | No default branch: an unregistered column and a registered column the table lacks both raise | 2 channels |
| 2 | A noise-floor table outlived the re-run of the records behind it; later scripts kept reading it | §10, T1 | The table is stamped over the runs; every reader, and the verdict step, refuses it after a re-run | 4 resolution, X1 |
| 3 | Analyses after a repair recorded the pre-repair commit, because the fix was uncommitted | §9 | Code state hashes the working tree; old records are refused instead of resumed | 3 runner, X1 |
| 4 | A leaderboard averaged each method over whichever cells it had finished | §3, §9, T4 | Coverage counts every planned run; resolution refuses a missing or failed run | 3 runner, 4 resolution |
| 5 | A pooled bootstrap resampled rows, counting each trial once per seed; intervals too narrow | §11 | Resampling is clustered on the unit that repeats | 4 resolution |
| 6 | A model was chosen on the test set and then bootstrapped on the same rows | §11 | `select` uses validation only; `oracle_select` is labelled and cannot decide a confirmatory claim | 5 rules, 6 claims |
| 7 | The memorization test read each drop on its own and declared recall | §11, T28b | The null-input gate refuses the rule; `difference_of_drops` compares the drops | 5 rules |
| 8 | Findings were published with no registered decision rules | §12 | A claim registered after the first run cannot be resolved | 6 claims |

## Defects the guards caught while this package was built

- The fixture hash left out the source tables, so changed source data reused a stale fixture (caught by a unit 2 test).
- `k_of_n_cells` counted point differences above a single-method floor and claimed an effect on 4 of 20 no-effect draws (caught by the null-input gate).
- The null-input gate allowed a fixed 3 effects, so a 3-draw suite passed a rule that always claims an effect (caught by a gate test).
