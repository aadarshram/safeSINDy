# Results

Every file here is produced by exactly one script; re-running that script
regenerates it (note non-determinism below). Grouped by what question each
answers, in the order the write-ups ([docs/sindy_rl.md](../docs/sindy_rl.md),
[docs/sindy_safe_rl.md](../docs/sindy_safe_rl.md)) raise them.

## Checkpoint 1 -- does the distilled sparse policy match the network?

| File | From | Shows |
|---|---|---|
| `one_episode.txt` | `scripts/run_one_episode.py` | The fitted polynomial law, and one episode's return for each controller: network 839.3, sparse policy 850.2 -- the distillation matches (slightly exceeds) the network on its own training distribution. |
| `nn_episode.mp4`, `sindy_episode.mp4` | `scripts/run_one_episode.py` | The two episodes behind that comparison. |

## Checkpoint 2 -- does the filter actually prevent violations?

| File | From | Shows |
|---|---|---|
| `validate_filter_summary.txt` | `scripts/validate_filter.py` (default: bilateral, gamma=0.5/0.5, 50 episodes) | The headline validation: 0/50 right-side violations, a handful of left-side violations (rail-edge-anchored left limit, looser than the right), mean return ~189. Run-to-run violation counts vary by a few episodes either way -- `env.reset()` is not seeded deterministically in this environment (confirmed empirically), so exact per-episode trajectories never repeat. |
| `demo/trace_unconstrained.png`, `demo/trace_constrained.png` | `scripts/demo_comparison.py` | Position traces for one representative violating unfiltered episode and one representative safe filtered episode, with the barrier lines and the violation region shaded. |
| `demo/frame_unconstrained.png`, `demo/frame_constrained.png`, `demo/video_unconstrained.mp4`, `demo/video_constrained.mp4` | `scripts/demo_comparison.py` | Rendered frames/videos for the same two episodes -- the actual overshoot (unfiltered) and the filter holding at the barrier (filtered). |
| `demo/paired_demo_data.npz` | `scripts/demo_comparison.py` | Raw observations backing the two traces/frames above (`obs_u`, `obs_c`, plus returns and extremes). |
| `gamma_sweep_mc.csv` | `scripts/gamma_sweep_mc.py` (9 `(gamma_cart, gamma_pole)` pairs, 15 episodes each) | The practical, non-formal search behind the gamma=0.5/0.5 choice: mean return, violation count, and worst-case extent for each pair. Picked because no gamma is SOS-certified safe for the pole (below), so the deployed value has to come from somewhere -- this is where. No pair in this grid eliminates violations outright (15 short episodes per pair is a coarse search, not a certificate); `(0.4, 0.4)` and `(0.7, 0.7)` happened to see 0/15 in this run, `(0.5, 0.5)` saw 5/15, matched gains below 0.5 generally fared better than mismatched or larger ones. This script picks a practical default, it does not replace `validate_filter.py`'s larger-sample validation of that default. |

## Formal verification -- can any gamma (or barrier position) be proven safe?

Produced by `sos/*.py`, in the `sos_verify` conda environment (needs `cvxpy`
with semidefinite support; see [sos/README.md](../sos/README.md)).

| File | From | Shows |
|---|---|---|
| `sos/dense_sweep_1.5.csv`, `figures/sos_dense_sweep.png` | `sos/dense_sweep.py`, plotted by `sos/plot_dense_sweep.py` | 1D sweep, 52 gamma values at the default barrier (x_limit=1.5): cart certifies SOS-safe up to about gamma=2.0; **the pole constraint certifies safe at 0 of the 52 points, in either region.** The centerpiece negative result. |
| `sos/barrier_offset_sweep.csv` | `sos/barrier_offset_sweep.py` | 2D sweep, 7 barrier positions x 14 gammas (98 points), degree-1 relaxation: same result generalized across barrier position, not just gamma -- 0/98 pole points certify. A few points return `solver_error`/`infeasible_inaccurate` rather than a clean verdict (a known CLARABEL numerical-precision limitation, not evidence either way). |
| `sos/deg2_barrier_offset_sweep.csv` | `sos/deg2_sweep.py` | The same 98-point grid, degree-2 relaxation with the SCS solver: **all 98 pole points now cleanly return `infeasible`**, no ambiguous verdicts. Confirms the degree-1 result is not a relaxation or solver artifact. |

## Model uncertainty -- does the SINDy ensemble's own fitted uncertainty predict its real error?

Produced by `scripts/model_uncertainty.py` (ensemble trimming and the
variance formula live in `safesindy.uncertainty`, reused by the filter's
adaptive-inflation code path -- not deployed, see
[docs/sindy_safe_rl.md](../docs/sindy_safe_rl.md) Section 10).

| File | Shows |
|---|---|
| `figures/uncertainty_tracking.png`, `figures/uncertainty_episode.npz` | One real rollout: true vs. SINDy-predicted rightmost body extent, with a +-5*sigma predicted band from the covariance formula (Eq. 2). The true trajectory stays inside the band on 951/1000 steps. |
| `figures/uncertainty_correlation.png`, `figures/uncertainty_scatter.npz` | 5000 real transitions: true one-step error (median ~0.01mm, max ~0.76mm) vs. two uncertainty estimators -- naive trimmed-ensemble spread (correlation ~0.81 with true error, but badly uncalibrated: a 1-sigma interval covers only ~5% of true errors) and the SINDy-RL covariance formula (correlation ~0.77, but well-calibrated: a 1-sigma interval covers ~41%, close to the Gaussian ideal). The formula is the one worth using as an actual margin; the naive spread is not, despite similar correlation. |

## Reproducing all of this

```bash
python scripts/run_one_episode.py
python scripts/validate_filter.py --n-episodes 50
python scripts/demo_comparison.py
python scripts/gamma_sweep_mc.py --n-episodes 15
python scripts/model_uncertainty.py
cd sos && python dense_sweep.py && python barrier_offset_sweep.py && python deg2_sweep.py && python plot_dense_sweep.py
```

`env.reset()` is not seeded deterministically in this cart-pole environment
(confirmed empirically), so any script that rolls real episodes will give
slightly different violation counts, returns, and representative episodes
each time -- the SOS verification results (deterministic, no environment
rollouts) will not change.
