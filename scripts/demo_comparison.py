"""
Unconstrained-vs-filtered demo used in the presentation (slide: "Demonstration:
unconstrained versus filtered").

Finds one representative unconstrained rollout that genuinely violates the
right barrier (non-vacuous: the unfiltered swing-up transiently overshoots it)
and one representative bilateral-filtered rollout that stays safe on both
sides, then renders the frames, trace plots, and videos used in the deck.

Usage:
    python scripts/demo_comparison.py [--out results/demo]
"""
import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safesindy.barrier import RAIL_LEFT, RAIL_RIGHT, body_left, body_right, right_barrier
from safesindy.barrier_view import render_barrier_frames
from safesindy.load import load_swingup
from safesindy.mirror import build_bilateral_barrier
from safesindy.rollout import _obs_from_reset, _unpack_step

GAMMA_CART, GAMMA_POLE = 0.5, 0.5
LEFT_LIMIT = -1.7
DT = 0.01


def build_bilateral(loaded, right_limit):
    return build_bilateral_barrier(
        loaded["dyn_model"].predict, loaded["policy"], right_limit, LEFT_LIMIT,
        gamma_cart=GAMMA_CART, gamma_pole=GAMMA_POLE, dt=DT,
    )


def run_until(env, nominal, bilateral, use_filter, right_limit, n_eps_try=40):
    """
    Search fresh rollouts (env.reset() is non-deterministic) for either:
      - unconstrained: the single worst (largest-overshoot) violating episode,
        confirming the barrier is non-vacuous; or
      - constrained: the first fully safe episode, as a representative run.
    """
    best = None
    for _ in range(n_eps_try):
        obs = _obs_from_reset(env.reset())
        state = np.asarray(obs, dtype=float).reshape(-1)
        observations = [state.copy()]
        br, bl, ret = [], [], 0.0
        for _step in range(1000):
            action = bilateral.compute_action(state) if use_filter else np.asarray(
                nominal.compute_action(state), dtype=np.float32
            ).reshape(-1)
            br.append(body_right(state[0], state[2]))
            bl.append(body_left(state[0], state[2]))
            obs, r, done, _ = _unpack_step(env.step(action))
            state = np.asarray(obs, dtype=float).reshape(-1)
            observations.append(state.copy())
            ret += float(r)
            if done:
                break
        br, bl = np.array(br), np.array(bl)
        max_r, min_l = br.max(), bl.min()
        violated = max_r > right_limit + 1e-6 or min_l < LEFT_LIMIT - 1e-6
        if use_filter and not violated:
            return observations, br, bl, ret, max_r, min_l
        if not use_filter and violated:
            score = max(max_r - right_limit, LEFT_LIMIT - min_l)
            if best is None or score > best[0]:
                best = (score, observations, br, bl, ret, max_r, min_l)
    if use_filter:
        return None
    return best[1:]


def plot_trace(tag, br, bl, ret, right_limit, out_dir):
    t = np.arange(len(br)) * DT
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, br, color="#1f6feb", lw=1.6, label="rightmost body extent")
    ax.plot(t, bl, color="#da3633", lw=1.6, label="leftmost body extent")
    ax.axhline(right_limit, color="#1f6feb", ls="--", lw=1.2, alpha=0.6, label="right barrier")
    ax.axhline(LEFT_LIMIT, color="#da3633", ls="--", lw=1.2, alpha=0.6, label="left barrier")
    viol_r, viol_l = br > right_limit + 1e-6, bl < LEFT_LIMIT - 1e-6
    if viol_r.any():
        ax.fill_between(t, right_limit, br, where=viol_r, color="#1f6feb", alpha=0.25)
    if viol_l.any():
        ax.fill_between(t, LEFT_LIMIT, bl, where=viol_l, color="#da3633", alpha=0.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.set_title(f"{'No filter' if tag == 'unconstrained' else 'CBF filter active'}  —  return = {ret:.1f}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    fig.savefig(out_dir / f"trace_{tag}.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/demo"))
    parser.add_argument("--check-idx", type=int, default=1150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loaded = load_swingup("data/agents/swingup/dyna", check_idx=args.check_idx)
    loaded["dyn_model"].set_median_coef_()
    env, nominal = loaded["real_env"], loaded["policy"]
    right_limit = right_barrier(RAIL_LEFT, RAIL_RIGHT)
    bilateral = build_bilateral(loaded, right_limit)

    print("searching for a representative unconstrained (violating) episode...", flush=True)
    obs_u, br_u, bl_u, ret_u, maxr_u, minl_u = run_until(env, nominal, bilateral, False, right_limit)
    print(f"  unconstrained: return={ret_u:.1f} max_body_right={maxr_u:.3f} min_body_left={minl_u:.3f}")

    print("searching for a representative constrained (safe) episode...", flush=True)
    obs_c, br_c, bl_c, ret_c, maxr_c, minl_c = run_until(env, nominal, bilateral, True, right_limit)
    print(f"  constrained:   return={ret_c:.1f} max_body_right={maxr_c:.3f} min_body_left={minl_c:.3f}")

    obs_u, obs_c = np.array(obs_u), np.array(obs_c)
    np.savez(
        args.out / "paired_demo_data.npz",
        obs_u=obs_u, br_u=br_u, bl_u=bl_u, ret_u=ret_u, maxr_u=maxr_u, minl_u=minl_u,
        obs_c=obs_c, br_c=br_c, bl_c=bl_c, ret_c=ret_c, maxr_c=maxr_c, minl_c=minl_c,
        right_limit=right_limit, left_limit=LEFT_LIMIT,
    )

    plot_trace("unconstrained", br_u, bl_u, ret_u, right_limit, args.out)
    plot_trace("constrained", br_c, bl_c, ret_c, right_limit, args.out)

    peak_idx = int(np.argmax(br_u))
    frame_u = render_barrier_frames([obs_u[peak_idx]], right_limit=right_limit, left_limit=LEFT_LIMIT)[0]
    imageio.imwrite(args.out / "frame_unconstrained.png", np.asarray(frame_u))
    print(f"  unconstrained peak overshoot at step {peak_idx}, body_right={br_u[peak_idx]:.3f}")

    mid_idx = min(500, len(obs_c) - 1)
    frame_c = render_barrier_frames([obs_c[mid_idx]], right_limit=right_limit, left_limit=LEFT_LIMIT)[0]
    imageio.imwrite(args.out / "frame_constrained.png", np.asarray(frame_c))

    frames_u = render_barrier_frames(obs_u[::2], right_limit=right_limit, left_limit=LEFT_LIMIT)
    imageio.mimwrite(args.out / "video_unconstrained.mp4", [np.asarray(f) for f in frames_u], fps=50)
    frames_c = render_barrier_frames(obs_c[::2], right_limit=right_limit, left_limit=LEFT_LIMIT)
    imageio.mimwrite(args.out / "video_constrained.mp4", [np.asarray(f) for f in frames_c], fps=50)

    print(f"saved all outputs to {args.out}/")


if __name__ == "__main__":
    main()
