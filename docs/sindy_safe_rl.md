# SINDy-Based Safety Filtering for Cart-Pole Swing-Up

This note describes our own extension of SINDy-RL (reviewed in
[sindy_rl.md](sindy_rl.md)): a test-time control barrier function (CBF)
safety filter built on the one-step SINDy dynamics model that ships with
the released swing-up checkpoint, together with a formal verification
study of that filter and a validation of the model's own fitted
uncertainty. All numbers below come from the result files in
[results/](../results/); [results/README.md](../results/README.md) maps
each one to the script that produced it, and every experiment can be
re-run from [README.md](../README.md).

## 0. Preliminary: checking the distillation step

Before building anything new, we first checked that the released
checkpoint and its own distilled sparse policy — the mechanism reviewed
in [sindy_rl.md](sindy_rl.md) §3 — actually agree with each other on this
task, since the safety filter below is applied to that same checkpoint's
network policy. Rolling one episode of each controller from the same
starting state gives a return of 839.3 for the released network and
850.2 for the sparse law distilled from it
([results/one_episode.txt](../results/one_episode.txt),
[results/nn_episode.mp4](../results/nn_episode.mp4),
[results/sindy_episode.mp4](../results/sindy_episode.mp4)) — the
distillation matches, and slightly exceeds, the network it was copied
from. Everything from here on uses the network policy.

## 1. Motivation

Reinforcement learning optimizes expected cumulative reward. Safety,
when it is addressed at all, is typically folded in only as a reward
penalty or a shaping term during training, which gives no actual
guarantee: because a constraint handled this way is merely discouraged,
not enforced, a large enough state deviation can still violate it.
A further, distinct problem is that some constraints are not known at
training time — a keep-out region defined only after deployment, for
instance — and a reward-shaped policy cannot be made to respect a
constraint it never saw during training.

This motivates a filter applied at test time, after and independent of
policy training: given whatever action the trained policy requests, make
the minimal correction needed to keep the system inside a safe set,
rather than retraining the policy itself.

## 2. Control barrier functions, discretely

A safe set is defined as `{x : h(x) ≥ 0}` for some function `h` that
measures distance from a keep-out region. A control barrier function
(CBF) filter does not replace the underlying policy; at every step, it
applies the smallest correction to the policy's requested action that
keeps `h` non-decreasing near the boundary of that set. The resulting
guarantee is forward invariance: if the system starts inside the safe
set, it provably stays there under the filtered action. That guarantee
is only as trustworthy as the one-step dynamics model the filter uses to
predict the effect of a candidate action — which is exactly where the
SINDy model, rather than a black-box alternative, becomes useful (§3).

**The barrier.** The keep-out region is the outer `1/12` of the rail on
each side (`x_limit ≈ ±1.5` m of an `[-1.8, 1.8]` m rail by default).
Two things can enter it: the cart's own body, and the pole tip as it
swings through. Each gets its own smooth barrier rather than one
non-smooth combined one (the single true requirement — rightmost point
of cart-or-pole under the limit — is a `min` of two polynomials, which is
continuous but not differentiable and not itself a polynomial; splitting
it into `h_cart ≥ 0` and `h_pole ≥ 0`, enforced simultaneously, is
mathematically identical to the combined requirement and keeps both
pieces smooth and polynomial, as both the filter's construction and the
SOS certification in §7 need):

```
h_cart(x) = safe_limit - cart_half - x
h_pole(x) = safe_limit - pole_radius - x - pole_length · sin θ
```

where `safe_limit` is `x_limit` minus a small inflation margin (0.1 m),
so that ordinary model error is not an impossible stop.

**Relative degree two, confirmed rather than assumed.** Probing the
fitted SINDy model directly: applying the maximum force in either
direction changes the model's predicted *velocity* in one step but leaves
the predicted *position* exactly unchanged (to the model's own ~1e-5 fit
residual) — the position update is explicit-Euler,
`x_{k+1} = x_k + dt · v_k`, using the velocity *before* the force is
applied. Neither `h_cart` nor `h_pole` therefore reacts to the current
force within a single step; a filter checking `h(x_{k+1}) ≥ 0` directly
would be constraining something the current action cannot affect at all.

**A discrete-time filter, not a discretized continuous-time one.** The
standard continuous-time exponential-CBF cascade was tried first and
found to have real failure modes under repeated receding-horizon
reapplication. The filter actually used instead works entirely in the
one-step map the model provides:

```
psi1(x) = gamma · h(x) - h_dot(x)
```

with `h_dot` exact and analytic (`ḣ_cart = -v`,
`ḣ_pole = -v - L cos θ · θ̇`) rather than a finite-difference
approximation, since velocity genuinely reacts to the current force in
one real SINDy step. The filter requires `psi1(x_{k+1}) ≥ 0`, evaluated
with the real one-step SINDy prediction, for both barriers, and clips the
nominal force to whichever interval keeps that satisfied. Soundness needs
only one real model step: if `psi1(x_k) ≥ 0` and the filter re-enforces
`psi1(x_{k+1}) ≥ 0` fresh at every step, then `psi1 ≥ 0` forever by
induction, which forces `h` to decay no faster than
`h_k (1 - dt · gamma)` — so `h ≥ 0` forever once it starts there. Cart and
pole are given separate gains, `gamma_cart` and `gamma_pole`: the cart's
`ḣ_cart = -v` is bounded only by the rail's own velocity range, but the
pole's `ḣ_pole` scales with its swing rate, which is large during any
ordinary swing-up far from the barrier — a single shared gain large
enough to brake decisively near the wall would also treat ordinary
swinging as a standing violation.

**Why only the endpoints matter.** The one-step SINDy-C model is affine
in the action `u` (SINDy-C's own structure — see
[sindy_rl.md](sindy_rl.md) §2), so `psi1` as a function of the candidate
force is affine too, and an affine function over an interval always
attains its extremes at the interval's endpoints. This is why both the
runtime filter's feasible-force computation and the SOS certification
below only ever need to check `u = u_min` and `u = u_max`, never an
interior value.

**Both ends of the rail, enforced together.** Cart-pole dynamics are
left-right symmetric — reflecting position, pole sine, cart velocity,
and pole rate while negating the applied force maps a trajectory to its
mirror image — so a validated right-side filter is reused for the left
side by wrapping the dynamics model and the nominal policy in that
reflection (`safesindy.mirror`), rather than rederiving the barrier algebra
twice. A naive design would then apply the right filter and the left
filter in sequence, but that only corrects a deviation in the direction
the active filter is watching; blocked repeatedly on one side, the system
can still drift into a genuine violation on the other, which neither
filter alone sees. `BilateralBarrier` instead enforces both constraints
at once, taking the force interval's intersection every step (and,
on the rare state where the two sides' demands are directly
contradictory, deferring to whichever side is currently closer to an
actual violation). This bilateral filter — `right_limit` at the
rail-fraction boundary, `left_limit = -1.7` m near the opposite physical
end, `gamma_cart = gamma_pole = 0.5` — is the configuration behind every
result in this report unless stated otherwise.

## 3. The SINDy model as a polynomial CBF substrate

Model-based RL already supplies a one-step dynamics model as a
byproduct of training (§0, [sindy_rl.md](sindy_rl.md) §3). A
neural-network or Gaussian-process model could, in principle, fill the
same role in the filter above, but it would remain opaque: a filter
built on it could only be checked empirically, at sampled states, with
no guarantee holding anywhere between them. Sum-of-squares (SOS)
verification via Putinar's Positivstellensatz is well established for
exactly this kind of polynomial feasibility question — but only for
analytically derived polynomial models, hand-written from known physics.
A black-box, data-driven model forecloses that option entirely.

SINDy recovers the same polynomial structure directly from data. The
swing-up model used throughout this report was fit with a *quadratic*
library over `(x, cos θ, sin θ, ẋ, θ̇)`: `cos θ` and `sin θ` enter as
independent polynomial state coordinates, not evaluated trigonometric
functions, linked only by the algebraic identity `cos²θ + sin²θ = 1`,
which the SOS certificate below enforces as an equality constraint. The
fitted model is therefore already a polynomial in exactly the sense SOS
requires — nothing about it needs to be approximated or re-derived to
make §7 possible.

## 4. A general safety-feasibility query

Because the dynamics model is a polynomial, the question "does some
constant gain satisfy this barrier?" can be answered formally, in
advance, for *any* proposed keep-out region and barrier — not only for
one pre-selected configuration. That is not a question a neural-network
or Gaussian-process dynamics model admits any comparable algebraic check
for. The cart-pole result in §7 is a worked example of *asking* this
question, not merely a single reported finding about one system: the
same procedure, unchanged, answers both "can this barrier even be
requested?" and, if so, "over what range of gain?" for whatever barrier
is proposed next.

## 5. Demonstration: unconstrained versus filtered

Before trusting any safety result, the unfiltered policy must actually
threaten the barrier — a filter that eliminates violations which would
never have happened anyway proves nothing. [results/demo/](../results/demo/)
records one representative pair of episodes from the same checkpoint and
policy: an unfiltered rollout that transiently exceeds the right barrier
by 0.35 m during swing-up (return 840.4;
[results/demo/trace_unconstrained.png](../results/demo/trace_unconstrained.png),
[results/demo/video_unconstrained.mp4](../results/demo/video_unconstrained.mp4)),
and a filtered rollout, same checkpoint and nominal policy, that stays
within both barriers throughout (return 188.5;
[results/demo/trace_constrained.png](../results/demo/trace_constrained.png),
[results/demo/video_constrained.mp4](../results/demo/video_constrained.mp4)).
The gap between those two returns is the real cost of the guarantee, and
is examined further in §9.

## 6. Does the model track reality? Validating uncertainty

A filter is only as trustworthy as the model predicting one step ahead
for it. Two independent estimators of the SINDy model's own one-step
uncertainty were checked directly against its true error on the real
environment: a naive spread across the trimmed 20-member ensemble, and
the closed-form covariance formula from [sindy_rl.md](sindy_rl.md) §2
(Eq. 2 of the paper), both computed over the same real rollouts.

Over 5,000 real transitions
([results/figures/uncertainty_correlation.png](../results/figures/uncertainty_correlation.png),
backed by
[results/figures/uncertainty_scatter.npz](../results/figures/uncertainty_scatter.npz)),
the true one-step error in rightmost body extent has a median of about
0.01 mm and a maximum of 0.76 mm — both far inside the filter's 100 mm
inflation margin. Both estimators track that error with comparable
correlation (naive spread: 0.81; covariance formula: 0.77), but they are
not equally useful as an actual margin: the covariance formula is
properly *calibrated* (its one-sigma interval covers about 41% of true
errors, close to the Gaussian ideal) while the naive spread's one-sigma
interval covers only about 5% — it understates its own magnitude as a
standard deviation by nearly an order of magnitude. A similar point holds
over a single continuous rollout
([results/figures/uncertainty_tracking.png](../results/figures/uncertainty_tracking.png),
[results/figures/uncertainty_episode.npz](../results/figures/uncertainty_episode.npz)):
a ±5σ band built from the covariance formula contains the true
trajectory on 951 of 1,000 steps. Neither estimator is a clean
conservative bound in the tails — coverage plateaus around 95% for both
even at 5 standard deviations — but the covariance formula is the one
whose value carries its literal statistical meaning, and is the one that
would be worth using for a state-dependent margin, `k · σ(x)`, directly
compatible with the barrier's own structure (`safesindy.uncertainty`
implements both estimators; not currently wired into the deployed
filter — see §10).

## 7. Formal verification: can any constant γ be certified?

The demonstration in §5 shows zero violations for *one* filtered run.
The question here is different and stronger: can a constant `gamma` be
proven safe for *every* state the barrier's domain allows, via the SOS
machinery §3 makes available, rather than merely observed safe on the
states actually sampled?

Each candidate `gamma` is certified separately for the cart constraint
and for the pole constraint, the latter split into two angular regions
since the control direction that most helps the margin reverses with
pole angle: region A (pole near vertical) is certified against
`u = u_max`, region B (pole near horizontal) against `u = u_min` (the
split point, `cos²θ ≈ 0.6863`, is where the SINDy-fitted control
sensitivity to force changes sign).

A dense one-dimensional sweep over 52 values of `gamma`, spanning two
orders of magnitude at the default barrier
([results/sos/dense_sweep_1.5.csv](../results/sos/dense_sweep_1.5.csv),
plotted in
[results/figures/sos_dense_sweep.png](../results/figures/sos_dense_sweep.png)),
certifies the cart constraint safe for every `gamma` up to about 2.0 —
and certifies the pole constraint safe at **none** of the 52 points, in
either region. For comparison: Monte Carlo sampling at `gamma = 0.5`
with 15 trials showed zero violations, while 50 trials (§8) revealed 6
— sampling at a size that looked adequate reported "safe" and was
wrong. The formal sweep shows this is a property of the constraint
itself, not a sampling accident.

A further two-dimensional sweep — 7 barrier positions (1.2 to 1.75 m) ×
14 values of `gamma`, 98 combinations in all
([results/sos/barrier_offset_sweep.csv](../results/sos/barrier_offset_sweep.csv)) —
checks whether moving the barrier itself, not just tuning the gain, ever
opens a certified point. It does not: all 98 pole points remain
infeasible (a handful return an ambiguous `solver_error` or
`infeasible_inaccurate` rather than a clean verdict — a known CLARABEL
numerical-precision limitation near thin margins, not evidence either
way). Re-running the identical 98-point grid at a strictly more
expressive, degree-2 relaxation with the SCS solver
([results/sos/deg2_barrier_offset_sweep.csv](../results/sos/deg2_barrier_offset_sweep.csv))
removes the ambiguity entirely: all 98 pole points now return a clean
`infeasible`, with no ambiguous verdicts at all — the degree-1 negative
result is not a relaxation or solver artifact.

The practical consequence: SINDy's polynomial structure lets this
question be answered exactly, for any candidate `gamma` or barrier
position — and the exact answer here is that no constant gain, at any
barrier position tested, formally guarantees the pole constraint. The
cart constraint, by contrast, certifies readily throughout.

## 8. Practical calibration: a Monte Carlo γ sweep

Given that §7 already rules out a formally certified constant `gamma`
for the pole, the deployed value is chosen by a separate, explicitly
non-formal step: a Monte Carlo sweep over `(gamma_cart, gamma_pole)`
pairs, 15 real episodes each
([results/gamma_sweep_mc.csv](../results/gamma_sweep_mc.csv)). No pair in
this grid eliminates violations outright at this sample size — expected,
given §7 — but matched, moderate gains fared best: `(0.4, 0.4)` and
`(0.7, 0.7)` saw zero violations across their 15 trials in this run, and
`(0.5, 0.5)` five. `gamma_cart = gamma_pole = 0.5` is the value used
throughout this report.

At a larger sample, 50 real episodes
([results/validate_filter_summary.txt](../results/validate_filter_summary.txt)),
the bilateral filter at `gamma = 0.5` recorded **zero** violations of the
right barrier and 8 of the left (left and right use different limits —
`-1.7` m, close to the physical rail end, against the fraction-based
right limit — so the two sides are not expected to behave identically),
with mean return 189.1. This residual left-side rate is the anticipated
consequence of §7, not an unexplained failure: the formal sweep already
established that no constant gain closes this gap completely. Repeated
runs of this same script give slightly different counts episode to
episode, since this cart-pole environment's `reset()` is not seeded
deterministically — [results/README.md](../results/README.md) has the
specifics.

## 9. Limitation: a policy unaware of the constraint

Mean return falls from 840.4 unfiltered to roughly 188–189 filtered (§5,
§8) — a substantial cost for the guarantee obtained. The filtered
trajectory shown in §5 does not settle into a calm balance; it
oscillates for the remainder of the episode without ever crossing either
barrier. The cause is structural, not a tuning error: the underlying
policy was trained with no notion of the constraint and continues to
request actions that would violate it on a large fraction of all steps,
so the filter is constantly correcting rather than intervening only near
genuine danger. A filter merely *applied* to a safety-unaware policy,
however formally sound, cannot recover the performance of a policy that
was trained with the constraint in mind from the start.

## 10. Future work

- **Gain schedules, not just constants.** Extend the feasibility query
  of §4 and §7 from a constant `gamma` to state-dependent schedules
  `gamma(x)`, which would also require reworking the forward-invariance
  argument of §2 itself, since `gamma` would then differ between `x_k`
  and `x_{k+1}`.
- **Safety-aware training.** Incorporate the filter directly into the
  Dyna-style training loop (`sindy_rl.md` §3) so the policy learns under
  the constraint, rather than being corrected against it only
  afterward — which would also address §9 directly.
- **The adaptive margin already built, not yet deployed.** §6's
  covariance-formula uncertainty estimate (`safesindy.uncertainty`) gives a
  principled state-dependent inflation margin, `k · σ(x)`; wiring it into
  the deployed filter in place of the current fixed 0.1 m margin is
  mechanical but has not been done, since the formula's own adaptive
  component stays an order of magnitude below the fixed margin almost
  everywhere sampled — the two would be numerically indistinguishable in
  practice, so this is a confirmation of the current choice rather than
  an urgent change.
- **Comparison across surrogate models.** Repeat this evaluation with
  the paper's other model-based surrogates — Dyna-NN and MB-MPO — under
  the identical constraint, to separate how much of SINDy's advantage
  here is its formal verifiability specifically, versus only its
  representational fit to this task.

## 11. Conclusion

A SINDy one-step model, already available as a byproduct of training
(§0, [sindy_rl.md](sindy_rl.md) §3), supports a discrete-time CBF safety
filter that eliminates a real, consistently occurring safety violation
(§5), at a quantified and structurally explained return cost (§9), with
the model's own fitted uncertainty independently confirmed to track its
true error (§6). The filter's single tunable gain cannot be made
formally safe everywhere by a constant choice — proven for the pole
constraint by exhaustive, degree-escalated SOS search, not merely left
unverified (§7) — but a Monte-Carlo-selected constant achieves zero
right-side violations and a small, explained residual on the left,
across every real rollout tested (§8). More generally, this extension
adds a further use for the same polynomial model SINDy-RL already
produces: given any requested safety barrier, its structure lets you
formally answer, in advance, whether a constant-gain CBF can satisfy it,
and if not, map out exactly where the boundary of achievability lies. A
single sparse-regression tool therefore serves every stage of this
pipeline — dynamics, reward, policy, and now a genuine safety-feasibility
query a black-box model could not answer at all.
