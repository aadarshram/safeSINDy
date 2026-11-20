"""
Practical (Monte Carlo) gamma sweep for the bilateral filter: tries several
(gamma_cart, gamma_pole) pairs and reports mean return and violation counts
for each, over real rollouts. This is the evidence behind the "practical
complement" slide, and the source of the gamma=0.5 choice later confirmed
at larger sample size by validate_filter.py.

Usage:
    python scripts/gamma_sweep_mc.py [--n-episodes 15]
"""
import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import csv
from pathlib import Path

import numpy as np

from safesindy.barrier import RAIL_LEFT, RAIL_RIGHT, body_left, body_right, right_barrier
from safesindy.load import load_swingup
from safesindy.mirror import build_bilateral_barrier
from safesindy.rollout import _obs_from_reset, _unpack_step

LEFT_LIMIT = -1.7
DEFAULT_PAIRS = [(0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.5, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 0.5)]


def run(env, nominal, dyn_model, right_limit, gc, gp, n_episodes):
    bilateral = build_bilateral_barrier(dyn_model.predict, nominal, right_limit, LEFT_LIMIT, gamma_cart=gc, gamma_pole=gp)
    rets, maxright, minleft = [], [], []
    for _ep in range(n_episodes):
        obs = _obs_from_reset(env.reset())
        state = np.asarray(obs, dtype=float).reshape(-1)
        ret, mr, ml = 0.0, -np.inf, np.inf
        for _step in range(1000):
            action = bilateral.compute_action(state)
            mr = max(mr, body_right(state[0], state[2]))
            ml = min(ml, body_left(state[0], state[2]))
            obs, r, done, _ = _unpack_step(env.step(action))
            ret += float(r)
            state = np.asarray(obs, dtype=float).reshape(-1)
            if done:
                break
        rets.append(ret)
        maxright.append(mr)
        minleft.append(ml)
    rets, maxright, minleft = np.array(rets), np.array(maxright), np.array(minleft)
    n_unsafe = int(np.sum(maxright > right_limit + 1e-6) + np.sum(minleft < LEFT_LIMIT - 1e-6))
    return rets.mean(), rets.std(), maxright.max(), minleft.min(), n_unsafe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=15)
    parser.add_argument("--check-idx", type=int, default=1150)
    parser.add_argument("--out", type=Path, default=Path("results/gamma_sweep_mc.csv"))
    args = parser.parse_args()

    loaded = load_swingup("data/agents/swingup/dyna", check_idx=args.check_idx)
    loaded["dyn_model"].set_median_coef_()
    env, nominal = loaded["real_env"], loaded["policy"]
    right_limit = right_barrier(RAIL_LEFT, RAIL_RIGHT)

    rows = []
    print(f"{'g_cart':>7} {'g_pole':>7} {'mean_ret':>9} {'std':>7} {'max_r':>7} {'min_l':>7} {'n_unsafe':>8}")
    for gc, gp in DEFAULT_PAIRS:
        mr, sr, mxr, mnl, nu = run(env, nominal, loaded["dyn_model"], right_limit, gc, gp, args.n_episodes)
        print(f"{gc:7.2f} {gp:7.2f} {mr:9.1f} {sr:7.1f} {mxr:7.3f} {mnl:7.3f} {nu:8d}", flush=True)
        rows.append((gc, gp, mr, sr, mxr, mnl, nu))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma_cart", "gamma_pole", "mean_return", "std_return", "max_body_right", "min_body_left", "n_unsafe"])
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
