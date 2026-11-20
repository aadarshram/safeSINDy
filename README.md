# safesindy

A SINDy-based control barrier function (CBF) safety filter, formally
verified via sum-of-squares (SOS) programming, built on the released
SINDy-RL cart-pole swing-up checkpoint. Two checkpoints:

1. **Policy distillation.** Copy the released neural-network swing-up
   policy into a sparse polynomial control law, and compare the two.
2. **Safety filter + formal verification.** Build a discrete-time control
   barrier function (CBF) filter on top of the released policy's own
   one-step SINDy dynamics model, to keep the cart-pole off a keep-out
   region at either end of the rail; validate it empirically; then use the
   fact that the SINDy model is a literal polynomial to ask, via
   sum-of-squares (SOS) verification, whether the filter's tunable gain
   can be made *formally* safe for the pole, for any choice of that gain
   or the barrier's position. Separately, check whether the SINDy
   ensemble's own fitted uncertainty predicts where the model's error
   actually is largest.

The SINDy-RL paper itself is reviewed in [docs/sindy_rl.md](docs/sindy_rl.md);
our own extension -- the safety filter, its formal verification, and the
uncertainty validation -- is written up in full, with every result
referenced, in [docs/sindy_safe_rl.md](docs/sindy_safe_rl.md). This file
covers the repo layout and how to reproduce every experiment; for what the
result *files* already in the repo show, see
[results/README.md](results/README.md).

## Repo structure

```
safesindy/      reusable library: barrier filter, dynamics loading, policy
              distillation, uncertainty estimation, plotting/video helpers
scripts/      experiment scripts (checkpoints 1 and 2, bilateral CBF filter)
sos/          formal (SOS) verification of the filter -- separate conda env
tests/        pytest unit tests for safesindy/
results/      experiment outputs; see results/README.md for what each is
data/         checkpoints and fitted policies (gitignored, downloaded locally)
docs/         the two checkpoint write-ups
```

`safesindy/` holds everything reusable across scripts: `barrier.py` (the
`HigherOrderBarrier` CBF filter itself), `mirror.py` (left-right reflection,
`BilateralBarrier`, `build_bilateral_barrier`), `uncertainty.py` (ensemble
trimming and the covariance-formula variance estimate), `load.py`/`fit.py`
(loading the released checkpoint, fitting the sparse policy),
`sparse_policy.py`, `rollout.py`, `barrier_view.py`/`trace_plot.py`
(rendering), and `run_record.py` (dated run folders). Every script in
`scripts/` is orchestration on top of these -- if you find yourself
copying barrier-construction code between scripts, it belongs in
`safesindy/` instead.

## Setup

From this directory, with the `safesindy` conda environment active:

```bash
bash scripts/install.sh
bash scripts/download_swingup.sh
```

This downloads the released swing-up checkpoint into `data/agents/`
(gitignored). The SOS verification tooling in `sos/` needs a **separate**
environment (`numpy>=2`, incompatible with the rest of the project's
`numpy==1.26.4` pin) -- see [sos/README.md](sos/README.md) for why and how
to set it up. Every mujoco-rendering script sets `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl` itself; you don't need to.

## Checkpoint 1: policy distillation

```bash
python scripts/fit_sparse_policy.py          # fits the sparse law, saves to data/sparse_policy/
python scripts/run_one_episode.py            # rolls both controllers, one episode each
```

## Checkpoint 2: safety filter

The filter (`safesindy.barrier.HigherOrderBarrier`, combined on both rail
ends by `safesindy.mirror.BilateralBarrier`) is the same object in every
script below; they differ only in what they measure.

**Validate the filter** -- the headline "0/50 right, 6/50 left violations,
mean return 188" numbers, at the final gains (gamma_cart = gamma_pole = 0.5,
both barriers active):

```bash
python scripts/validate_filter.py --n-episodes 50
```

`--side {both,right,left}`, `--gamma-cart`, `--gamma-pole`, `--x-limit`,
`--left-limit`, and `--policy {nn,sparse}` (bridging back to checkpoint 1)
all generalize this to any single-barrier configuration -- including the
earlier right-side-only / left-side-only runs from before the bilateral
filter existed; see the script's own docstring for the exact historical
invocations. Add `--video` to render the first episode with its barrier
margin traced underneath.

**How the gain was chosen** -- a Monte Carlo sweep over `(gamma_cart,
gamma_pole)` pairs, since no choice is SOS-certified safe for the pole
(below) and the gain had to be picked practically instead:

```bash
python scripts/gamma_sweep_mc.py --n-episodes 15
```

**Unconstrained vs. filtered demo** -- finds one representative violating
unfiltered rollout and one representative safe filtered rollout, and
renders the frames, traces, and videos used in the write-up:

```bash
python scripts/demo_comparison.py
```

**Does the SINDy model's own uncertainty track its real error?** -- one
rollout's tracking plot plus a 5000-transition correlation scatter,
comparing naive trimmed-ensemble spread against the SINDy-RL covariance
formula (Eq. 2):

```bash
python scripts/model_uncertainty.py
```

## Formal verification (SOS)

In the **`sos_verify`** conda environment (see
[sos/README.md](sos/README.md)), from inside `sos/`:

```bash
cd sos
python dense_sweep.py           # 1D: fine gamma sweep at the default barrier -- the main negative result
python barrier_offset_sweep.py  # 2D: gamma x barrier position, degree 1
python deg2_sweep.py            # same 2D grid, degree 2 + a more robust solver -- confirms the degree-1 result
python plot_dense_sweep.py      # renders results/figures/sos_dense_sweep.png from dense_sweep.py's output
```

Each writes its CSV into `results/sos/`.

# Acknowledgements
- This work is developed as part of the course project in the course EE6434: Data-Driven Control and Learning, taught by Prof. Puduru Vishwanatha Reddy, Fall 2026, IIT Madras.
- The extension of SINDy for safety filters is built over the pre-trained checkpoints and theory developed in the original SINDy-RL paper:
  ```@article{zolman2025sindy,
  title={SINDy-RL for interpretable and efficient model-based reinforcement learning},
  author={Zolman, Nicholas and Lagemann, Christian and Fasel, Urban and Kutz, J Nathan and Brunton, Steven L},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={10714},
  year={2025},
  publisher={Nature Publishing Group UK London}
}```