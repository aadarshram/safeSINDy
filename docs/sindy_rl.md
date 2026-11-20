# SINDy-RL: A Review

This note reviews Zolman, Lagemann, Fasel, Kutz, and Brunton, "SINDy-RL
for Interpretable and Efficient Model-Based Reinforcement Learning"
(2025), the paper this project builds on. It covers what the paper does
and why, not our own work — our extension, built on the released
cart-pole checkpoint, is described separately in
[sindy_safe_rl.md](sindy_safe_rl.md).

## 1. Motivation

Deep reinforcement learning has reached superhuman performance in
domains ranging from game-playing to autonomous drone racing to tokamak
plasma control, but it gets there through direct trial-and-error
interaction with the environment, typically on the order of millions of
samples. For a physical system or a high-fidelity simulator, that sample
requirement is the dominant cost. The paper's own numbers make the point
concretely: a single step of its 3D-airfoil CFD environment takes about
45 seconds, and running that simulator at all requires 75 GB of GPU
memory spread across four NVIDIA A100s. A policy trained this way is
also, once trained, a large opaque network — hard to certify formally
and hard to deploy on constrained hardware.

The standard remedy is model-based RL in the Dyna style: fit a surrogate
model of the environment and train the agent mostly inside that cheap
surrogate, querying the real environment only periodically to correct
it. In most prior Dyna-style work, the surrogate is itself a neural
network. That is cheaper to query than the real environment, but it
still needs substantial data to fit well, can be unstable over long
rollouts, and remains, itself, a black box. The question SINDy-RL asks
is whether the surrogate can be made both cheap to query and
analytically transparent at the same time.

## 2. Background: SINDy

SINDy (Sparse Identification of Nonlinear Dynamics) answers that
question by regressing a system's dynamics onto a library of candidate
functions — polynomials, trigonometric terms, and so on — and keeping
only the small subset of terms whose fitted coefficients are
non-negligible:

```
Ξ = argmin ‖Y − Θ(X)Ξ‖² + ‖Ξ‖₁
```

`Y` is the measured next-state data, `Θ(X)` the candidate library
evaluated on it, and `Ξ` the resulting sparse coefficient matrix. `Ξ`
*is* the model: compact enough to read off and reason about directly,
rather than a weight tensor that only produces numbers.

Two extensions matter for what follows. First, **SINDy-C** augments the
library with the control input `u`, so the regression runs over
(state, action) pairs rather than state alone — this is what makes the
method applicable to a controlled system instead of only an autonomous
one. Second, SINDy-RL fits its models in **discrete time**,
`x_{k+1} = f(x_k, u_k)`, directly, rather than fitting a continuous-time
model that would then need to be integrated forward. This is a
deployment choice rather than a modeling necessity: a discrete-time
model interfaces directly with an RL environment's `step` function, with
no integration scheme to get right.

### Ensemble SINDy and uncertainty quantification

A single SINDy fit is sensitive to noise, particularly with limited
data. SINDy-RL instead fits an *ensemble* of SINDy models on
bootstrapped subsets of the data and retains the ensemble's median
coefficients, which is substantially more robust. Because the model is
linear in its coefficients, this ensemble buys something a
neural-network ensemble does not: a closed-form, pointwise predictive
variance,

```
Var[y(x) | x] = Θ(x) Cov(ξ) Θ(x)ᵀ
```

(Eq. 2 of the paper), with `Cov(ξ)` estimated directly from the spread of
the ensemble's fitted coefficient vectors. This is a genuine uncertainty
map over the state space, available at no extra computational cost
beyond the ensemble fit itself — the paper uses it to visualize how
confident the model becomes in the regions the policy actually visits as
training proceeds (Figure 7); our own extension puts the same formula to
a different, quantitative use (see [sindy_safe_rl.md](sindy_safe_rl.md),
§6).

## 3. The SINDy-RL framework

The paper's central contribution is less a single technique than a
unifying observation: in a model-based RL loop, the dynamics, the
reward, and the policy are each, independently, candidates for
replacement by a sparse dictionary model. Each can be swapped in only
where it is actually needed.

The training loop (Dyna-style) runs as follows. An initial dataset is
collected under a default policy — random actions, or an untrained
network. An ensemble-SINDy dynamics model is fit to that dataset. The
policy is then trained via PPO for many iterations almost entirely
inside this cheap surrogate. Periodically, a small amount of fresh data
is collected from the real environment, the surrogate is refit, and the
cycle repeats. The real environment is touched only for that periodic
correction — everything else happens inside the polynomial surrogate.

Where the reward is not a direct function of the observed state — the
paper's fluid-flow tasks measure only a sparse set of flow sensors, not
the quantity (e.g., drag) actually being optimized — a second dictionary
model is fit to predict reward from (observation, action) pairs, using
the identical regression machinery. For the cylinder task this yields an
explicit quadratic surface in (lift, lift-rate) whose minimum visibly
shifts with the applied control, recovering a directly interpretable
statement of how actuation reduces drag.

Finally, after a neural-network policy has been trained, the same
regression procedure is applied a third time, now to the policy itself:
behavior cloning distills the trained network into a sparse dictionary
law, the network serving as the expert and the fit matching its chosen
actions. On the Pinball task, this reduces 5,000+ network parameters to
66 polynomial coefficients — and the distilled law generalized *more*
reliably than the network it was copied from, while producing smoother
control.

## 4. Case study: cart-pole swing-up

The paper's primary worked example — and the one this project builds
its own extension on — is the classic cart-pole swing-up task. The
system starts at the stable, hanging-down equilibrium and must swing up
to stabilize at the unstable, upright equilibrium. The state is
`(x, cos θ, sin θ, ẋ, θ̇)` — cart position, the pole angle's cosine and
sine (rather than the angle itself, so the state stays smooth through a
full rotation), cart velocity, and pole angular rate — and the action is
a single horizontal force on the cart. The reward is a product of four
bounded tolerance terms, one per control objective (pole upright, cart
centered, force small, pole rate small), each in `[0, 1]` and equal to 1
exactly at its target.

## 5. Results

Four methods are compared on swing-up: baseline PPO (model-free, no
surrogate); Dyna-NN (the identical Dyna loop with a neural-network
surrogate in place of SINDy); MB-MPO (a standard model-based
meta-policy-optimization baseline); and SINDy-RL itself, both with a
quadratic library and, as a negative control, with a linear one too
restrictive to capture the true dynamics. SINDy-RL with the quadratic
library reaches baseline PPO's final performance using roughly **100×**
fewer real-environment interactions, and outperforms both model-based
baselines at matched data budgets.

The same approach scales to a realistic, CFD-class problem: a 3D airfoil
task mitigating a large incoming gust by jet blowing/suction at
Re = 1000, the same expensive environment the motivation section opened
with. SINDy-RL collects experience roughly 10,000× faster than the full
solver, and needed 14 hours of training to match what otherwise took 185
hours directly inside the full-order solver.

Generalization is checked on the Pinball task — stabilizing the wake
behind three cylinders — training exclusively at Reynolds number 100 and
evaluating at 150, 250, and 350, where the flow transitions from
quasi-periodic to chaotic. The distilled dictionary policy extrapolates
more reliably than the network it was derived from, and both outperform
the baseline, despite neither having trained on the harder regimes.

The ensemble's uncertainty estimate (§2) is also tracked over the course
of training itself: early on, the agent's exploration stays confined to
a narrow band near the initial condition, leaving the rest of the state
space high-variance; as training proceeds, the trajectory the agent
actually follows increasingly passes through low-variance regions — the
model becomes confident precisely where it is subsequently used.

The cylinder environment is the paper's one notable counterexample to
distillation: the network there learns a bang-bang switching law, and a
smooth polynomial cannot represent that discontinuity. The polynomial
copy reduces drag by about 3.7%, against roughly 8.7% for the network it
was distilled from.

## 6. Summary

A single sparse-regression framework yields interpretable surrogates for
the dynamics, the reward, and the policy, used independently and only
where needed. Training occurs almost entirely inside the cheap
surrogate, cutting required real-environment interaction by orders of
magnitude. And because the dynamics surrogate is an explicit,
low-dimensional polynomial rather than a black box, it admits a kind of
analysis a neural-network or Gaussian-process model does not: formal,
algebraic reasoning about what the model guarantees, rather than only
empirical querying of it.

That last point is the one this project picks up. SINDy-RL produces, as
a direct byproduct of ordinary training, both a trained policy and an
explicit polynomial model of the plant's dynamics — at no cost beyond
what the training loop already pays. The natural next question, which
the paper itself does not pursue, is what further use that polynomial
model supports once training is done. [sindy_safe_rl.md](sindy_safe_rl.md)
takes up that question for one specific and practically important use:
test-time safety.

## Reference

Zolman, N., Lagemann, C., Fasel, U., Kutz, J. N., Brunton, S. L. "SINDy-RL
for Interpretable and Efficient Model-Based Reinforcement Learning."
2025. Figures and equations cited above (Figures 1–7, Eq. 2, Algorithm 1,
§§1–3.5) refer to this paper.
