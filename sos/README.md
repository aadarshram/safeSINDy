# SOS verification tooling

Formal verification of the discrete-time CBF's `gamma` against the fitted
SINDy dynamics (checkpoint 1150, median ensemble coefficients), via
sum-of-squares (SOS) semidefinite programming -- the thing the dynamics
being a literal polynomial (a SINDy fit, not a neural-net or GP surrogate)
makes possible at all. `dense_sweep.py` is the primary result: a dense,
one-dimensional sweep over `gamma` that runs the actual SOS certificate
itself (not a proxy for it) for both barriers. `barrier_offset_sweep.py`
and `deg2_sweep.py` extend the same check to a 2D grid over barrier
position as well as `gamma`, at two relaxation degrees. The write-up with
all three results in context is
[docs/sindy_safe_rl.md](../docs/sindy_safe_rl.md) Section 7; the Monte
Carlo search used instead to pick a practical deployment value (since no
constant `gamma` is SOS-certified safe for the pole) lives outside this
folder, in `scripts/gamma_sweep_mc.py`, which runs in the main `safesindy`
environment against the real environment rather than the SOS solver.

## Separate environment required

These scripts need `cvxpy` with semidefinite support, which requires
`numpy>=2`. The rest of this project (`pysindy`, `scikit-learn`, `mujoco`,
`ray`/`rllib`) is pinned to `numpy==1.26.4` and breaks under numpy 2 (a
scikit-learn binary-incompatibility crash, confirmed). The two requirement
sets cannot coexist in one conda env.

Run everything in this folder from a **separate** environment:

```bash
conda create -n sos_verify python=3.11 -y
conda activate sos_verify
pip install cvxpy SumOfSquares
```

There is no automated bridge between the two environments. The SINDy
coefficients used in `pole_cert.py`/`sos_sweep.py` were extracted once, by
hand, from the loaded model in the `safesindy` env (median of the 20-member
ensemble, checkpoint 1150) and hardcoded as Python floats. If the model is
retrained, a different checkpoint is used, or the nominal policy changes,
these numbers go stale silently -- there is no error, the result is just
quietly about a model that no longer exists. Re-extracting them (and
ideally scripting the export instead of hand-copying) is the first thing
to fix before trusting this tooling on a different checkpoint.

## Files

Library code, imported by the experiment scripts below:

- `manual_sos.py` -- generic Gram-matrix / Putinar-style SOS certificate
  via `cvxpy`, independent of cart-pole. `deg` is the half-degree bound of
  each multiplier's monomial basis (a proof-strategy parameter), not the
  degree of the dynamics model itself, which is fixed by the SINDy fit.
- `pole_cert.py` -- the pole's one-step model and its two-region
  certificate (the region split is derived from where the control
  sensitivity to force changes sign, `cos^2(theta) ~= 0.6863`); also
  exposes `certify_region()` directly, with a `deg` and `solver` argument,
  used by `deg2_sweep.py` for the degree-2 confirmation.
- `sos_sweep.py` -- the cart's one-step model plus `cart_certify()` and
  `pole_certify_region()`, the default degree-1 wrappers every sweep script
  below is built on.

Experiment scripts, each writing its result into `results/sos/` (or
`results/figures/` for the plot):

- `dense_sweep.py` -- **primary result.** 52 values of `gamma` at the
  default barrier position: certifies the cart constraint safe up to
  `gamma ~= 2.0`; certifies the pole constraint infeasible at every one of
  the 52 points, in both regions.
- `barrier_offset_sweep.py` -- 2D: 7 barrier positions x 14 values of
  `gamma` (98 points), degree-1 relaxation. Checks whether moving the
  barrier itself, not just the gain, ever opens a certified-safe point for
  the pole. It does not; a few points return `solver_error` or
  `infeasible_inaccurate` rather than a clean verdict (a CLARABEL
  numerical-precision limitation near thin margins, not evidence of
  feasibility there).
- `deg2_sweep.py` -- the identical 98-point grid, re-certified at a
  degree-2 relaxation with the SCS solver (more numerically robust than
  CLARABEL at this degree). All 98 pole points now return a clean
  `infeasible`, with none of degree-1's ambiguous verdicts -- confirms the
  degree-1 negative result is not a relaxation or solver artifact.
- `plot_dense_sweep.py` -- renders `dense_sweep.py`'s CSV as
  `results/figures/sos_dense_sweep.png`.

Superseded scratch scripts used while building `manual_sos.py`, and an
earlier, purely Monte Carlo (non-formal) gamma search from before the
dense/2D sweeps existed, have been moved to `trash/sos/` rather than kept
here.
