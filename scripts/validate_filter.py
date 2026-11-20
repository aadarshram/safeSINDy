"""
Validate the discrete-time CBF safety filter against the real environment.

One script, one barrier configuration per run, built from whatever limits
and gains you pass in -- the final deployed config (bilateral, both rail
ends, gamma_cart = gamma_pole = 0.5) is just the default.

This replaces two earlier, narrower scripts:
  - the original single-sided filter runner (right-only or left-only,
    gamma_pole=8.0, swept over x_limit in {1.2 .. 1.75} -- the source of
    the per-position table in docs/sindy_safe_rl.md Section 8); and
  - a bilateral-only validator at the final gamma.
Both are still reachable here via --side and --gamma-*; see "Reproducing
the historical single-side results" below.

Usage:
    python scripts/validate_filter.py [--side both|right|left] [--n-episodes 50]
        [--gamma-cart 0.5] [--gamma-pole 0.5] [--x-limit X] [--left-limit X]
        [--policy nn|sparse] [--video]

Reproducing the historical single-side results (docs/sindy_safe_rl.md Section 8,
gamma_pole=8.0 era, before the bilateral filter and the final gamma=0.5 choice):
    python scripts/validate_filter.py --side right --gamma-cart 0.5 --gamma-pole 8.0 \\
        --x-limit 1.4 --n-episodes 15   # repeat for 1.5, 1.65
    python scripts/validate_filter.py --side left --gamma-cart 0.5 --gamma-pole 8.0 \\
        --left-limit -0.35 --n-episodes 15
"""
import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from datetime import datetime
from pathlib import Path

import numpy as np

from safesindy.barrier import RAIL_LEFT, RAIL_RIGHT, HigherOrderBarrier, body_left, body_right, right_barrier
from safesindy.barrier_view import render_barrier_frames
from safesindy.load import load_swingup
from safesindy.mirror import MirroredBarrier, ReflectedPolicy, build_bilateral_barrier, reflected_step_fn
from safesindy.rollout import _obs_from_reset, _unpack_step
from safesindy.run_record import RunRecord, make_run_dir, write_episode_video
from safesindy.sparse_policy import load_sparse_policy
from safesindy.trace_plot import frames_with_trace

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEFT_LIMIT = -1.7


def build_barrier(side, step_fn, nominal, right_limit, left_limit, gamma_cart, gamma_pole, inflation, dt):
    if side == "right":
        return HigherOrderBarrier(
            step_fn=step_fn, nominal=nominal, x_limit=right_limit,
            inflation=inflation, dt=dt, gamma_cart=gamma_cart, gamma_pole=gamma_pole,
        )
    if side == "left":
        inner = HigherOrderBarrier(
            step_fn=reflected_step_fn(step_fn), nominal=ReflectedPolicy(nominal), x_limit=-left_limit,
            inflation=inflation, dt=dt, gamma_cart=gamma_cart, gamma_pole=gamma_pole,
        )
        return MirroredBarrier(inner)
    return build_bilateral_barrier(
        step_fn, nominal, right_limit, left_limit,
        gamma_cart=gamma_cart, gamma_pole=gamma_pole, inflation=inflation, dt=dt,
    )


def run_episode(env, barrier, steps):
    obs = _obs_from_reset(env.reset())
    observations = [np.asarray(obs, dtype=float).reshape(-1)]
    actions, rewards = [], []
    total = 0.0
    for _step in range(steps):
        action = barrier.compute_action(obs)
        obs, reward, done, _info = _unpack_step(env.step(action))
        flat = np.asarray(obs, dtype=float).reshape(-1)
        observations.append(flat)
        actions.append(np.asarray(action, dtype=float).reshape(-1))
        rewards.append(float(reward))
        total += float(reward)
        if done:
            break
    observations = np.asarray(observations)
    max_right = float(body_right(observations[:, 0], observations[:, 2]).max())
    min_left = float(body_left(observations[:, 0], observations[:, 2]).min())
    return {
        "return": total, "max_right": max_right, "min_left": min_left,
        "observations": observations, "actions": np.asarray(actions, dtype=float), "rewards": np.asarray(rewards),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["nn", "sparse"], default="nn")
    parser.add_argument("--trial-dir", type=Path, default=ROOT / "data" / "agents" / "swingup" / "dyna")
    parser.add_argument("--sparse-path", type=Path, default=ROOT / "data" / "sparse_policy" / "cart_swingup_1150.json")
    parser.add_argument("--check-idx", type=int, default=1150)
    parser.add_argument("--side", choices=["both", "right", "left"], default="both")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--gamma-cart", type=float, default=0.5)
    parser.add_argument("--gamma-pole", type=float, default=0.5)
    parser.add_argument("--x-limit", type=float, default=None, help="Right barrier position (default: fraction 11/12 of the rail).")
    parser.add_argument("--left-limit", type=float, default=DEFAULT_LEFT_LIMIT, help="Left barrier position (negative).")
    parser.add_argument("--inflation", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--video", action="store_true", help="Render the first episode with its barrier trace.")
    parser.add_argument("--out", type=Path, default=Path("results/validate_filter_summary.txt"))
    args = parser.parse_args()

    loaded = load_swingup(args.trial_dir, check_idx=args.check_idx)
    loaded["dyn_model"].set_median_coef_()
    env = loaded["real_env"]
    nominal = loaded["policy"] if args.policy == "nn" else load_sparse_policy(args.sparse_path)

    right_limit = args.x_limit if args.x_limit is not None else right_barrier(RAIL_LEFT, RAIL_RIGHT)
    barrier = build_barrier(
        args.side, loaded["dyn_model"].predict, nominal, right_limit, args.left_limit,
        args.gamma_cart, args.gamma_pole, args.inflation, args.dt,
    )

    runs = []
    for ep in range(args.n_episodes):
        result = run_episode(env, barrier, args.steps)
        runs.append(result)
        r_viol = args.side != "left" and result["max_right"] > right_limit + 1e-6
        l_viol = args.side != "right" and result["min_left"] < args.left_limit - 1e-6
        flag = "  VIOLATION" if r_viol or l_viol else ""
        print(f"ep{ep}: ret={result['return']:.1f} max_right={result['max_right']:.3f} min_left={result['min_left']:.3f}{flag}", flush=True)

    rets = np.array([r["return"] for r in runs])
    maxright = np.array([r["max_right"] for r in runs])
    minleft = np.array([r["min_left"] for r in runs])
    n_r = int(np.sum(maxright > right_limit + 1e-6)) if args.side != "left" else 0
    n_l = int(np.sum(minleft < args.left_limit - 1e-6)) if args.side != "right" else 0
    summary = (
        f"SUMMARY side={args.side} policy={args.policy} n_eps={args.n_episodes} "
        f"gamma_cart={args.gamma_cart} gamma_pole={args.gamma_pole} "
        f"right_limit={right_limit:.3f} left_limit={args.left_limit:.3f}\n"
        f"mean_ret={rets.mean():.1f} std={rets.std():.1f}\n"
        f"right violations: {n_r}/{args.n_episodes}  left violations: {n_l}/{args.n_episodes}\n"
        f"worst right overshoot: {maxright.max() - right_limit:.3f}  "
        f"worst left overshoot: {args.left_limit - minleft.min():.3f}\n"
    )
    print("\n" + summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary)

    if args.video:
        run_dir = make_run_dir(ROOT / "results" / "filter_videos", f"validate_{args.side}_{args.policy}", datetime.now())
        record = RunRecord(run_dir)
        (run_dir / "summary.txt").write_text(summary)
        obs_arr = runs[0]["observations"]
        state_k, state_k1 = obs_arr[:-1], obs_arr[1:]
        frames = render_barrier_frames(state_k1)
        h = right_limit - body_right(state_k1[:, 0], state_k1[:, 2])
        h0, psi1_next = barrier.margins(state_k, state_k1)
        frames = frames_with_trace(frames, runs[0]["rewards"], h, plot_height=224, psi1=h0, psi2=psi1_next)
        video_path = write_episode_video(run_dir / "episode.mp4", frames)
        record.save_trajectory(obs_arr, runs[0]["actions"], runs[0]["rewards"])
        print(f"video: {video_path}")


if __name__ == "__main__":
    main()
