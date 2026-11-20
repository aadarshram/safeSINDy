"""
Model-uncertainty validation for the SINDy swing-up model (the two figures on
the "Does the SINDy model reflect the real dynamics?" slide): a one-episode
tracking plot of true vs. predicted rightmost body extent with a predicted
uncertainty band, and a correlation scatter comparing two uncertainty
estimators (naive trimmed-ensemble spread vs. the SINDy-RL covariance
formula, Eq. 2) against true one-step error, over many real transitions.

Ensemble trimming and the covariance-formula variance are both reusable
module code (safesindy.uncertainty): outliers are excluded by a MAD rule over
a real rollout, not a hardcoded index set -- typically ~4 of 20 members,
see docs/sindy_safe_rl.md Section 6.

Usage:
    python scripts/model_uncertainty.py [--n-episode-steps 1000] [--n-scatter-steps 5000] [--out results/figures]
"""
import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safesindy.barrier import body_right
from safesindy.load import load_swingup
from safesindy.rollout import _obs_from_reset, _unpack_step
from safesindy.uncertainty import VarianceInflation, trim_ensemble


def project(raw):
    out = np.array(raw, dtype=float, copy=True)
    norm = max(np.linalg.norm(out[1:3]), 1e-12)
    out[1:3] = out[1:3] / norm
    return out


def collect_trim_rollout(env, policy, n_steps):
    """One long rollout's (state, action) pairs, for trim_ensemble's outlier check."""
    states, actions = [], []
    obs = _obs_from_reset(env.reset())
    state = np.asarray(obs, dtype=float).reshape(-1)
    for _step in range(n_steps):
        action = np.asarray(policy.compute_action(state), dtype=np.float32).reshape(-1)
        states.append(state.copy())
        actions.append(action.copy())
        nxt, _r, done, _ = _unpack_step(env.step(action))
        state = np.asarray(nxt, dtype=float).reshape(-1)
        if done:
            obs = _obs_from_reset(env.reset())
            state = np.asarray(obs, dtype=float).reshape(-1)
    return states, actions


def collect_episode(env, policy, median_coef, library, std_fn, n_steps):
    true_ext, pred_ext, std_ext = [], [], []
    obs = _obs_from_reset(env.reset())
    state = np.asarray(obs, dtype=float).reshape(-1)
    true_ext.append(body_right(state[0], state[2]))
    for _step in range(n_steps):
        action = np.asarray(policy.compute_action(state), dtype=np.float32).reshape(-1)
        theta = library.transform(np.concatenate([state, action]).reshape(1, -1)).reshape(-1)
        median_pred = project(median_coef @ theta)

        true_next, _r, done, _ = _unpack_step(env.step(action))
        true_next = np.asarray(true_next, dtype=float).reshape(-1)

        true_ext.append(body_right(true_next[0], true_next[2]))
        pred_ext.append(body_right(median_pred[0], median_pred[2]))
        std_ext.append(std_fn(state, action[0]))

        state = true_next
        if done:
            break
    return np.array(true_ext[1:]), np.array(pred_ext), np.array(std_ext)


def collect_scatter(env, policy, median_coef, library, std_fn, trimmed_coefs, n_steps):
    true_errs, naive_std, cov_std = [], [], []
    obs = _obs_from_reset(env.reset())
    state = np.asarray(obs, dtype=float).reshape(-1)
    for _step in range(n_steps):
        action = np.asarray(policy.compute_action(state), dtype=np.float32).reshape(-1)
        theta = library.transform(np.concatenate([state, action]).reshape(1, -1)).reshape(-1)

        median_pred = project(median_coef @ theta)
        preds_trimmed = np.array([project(trimmed_coefs[i] @ theta) for i in range(len(trimmed_coefs))])
        ext_members = body_right(preds_trimmed[:, 0], preds_trimmed[:, 2])

        true_next, _r, done, _ = _unpack_step(env.step(action))
        true_next = np.asarray(true_next, dtype=float).reshape(-1)

        true_ext = body_right(true_next[0], true_next[2])
        pred_ext = body_right(median_pred[0], median_pred[2])

        true_errs.append(abs(true_ext - pred_ext))
        naive_std.append(float(ext_members.std()))
        cov_std.append(std_fn(state, action[0]))

        state = true_next
        if done:
            obs = _obs_from_reset(env.reset())
            state = np.asarray(obs, dtype=float).reshape(-1)
    return np.array(true_errs), np.array(naive_std), np.array(cov_std)


def plot_tracking(true_ext, pred_ext, std_ext, out_dir, k=5, dt=0.01):
    t = np.arange(len(true_ext)) * dt
    band_lo, band_hi = pred_ext - k * std_ext, pred_ext + k * std_ext
    miss = (true_ext < band_lo) | (true_ext > band_hi)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(t, band_lo, band_hi, color="#2471a3", alpha=0.25, label=rf"predicted uncertainty band ($\pm{k}\sigma$)")
    ax.plot(t, pred_ext, color="#2471a3", lw=1.0, label="SINDy one-step prediction")
    ax.plot(t, true_ext, color="#c0392b", lw=1.2, label="true environment", alpha=0.9)
    if miss.any():
        ax.scatter(t[miss], true_ext[miss], color="#c0392b", s=10, zorder=5, label=f"true value outside band ({miss.sum()}/{len(t)} steps)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rightmost body extent (m)")
    ax.set_title("SINDy one-step prediction vs. true environment, one real rollout")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_tracking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved uncertainty_tracking.png, missed {miss.sum()}/{len(t)}")


def plot_correlation(true_err, naive_std, cov_std, out_dir):
    corr_naive = np.corrcoef(true_err, naive_std)[0, 1]
    corr_cov = np.corrcoef(true_err, cov_std)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    axes[0].scatter(naive_std, true_err, s=6, alpha=0.3, color="#7f8c8d")
    axes[0].set_title(f"Naive trimmed-ensemble spread\ncorrelation = {corr_naive:.2f}")
    axes[0].set_xlabel("predicted std (m)")
    axes[0].set_ylabel("true one-step error (m)")

    axes[1].scatter(cov_std, true_err, s=6, alpha=0.3, color="#2471a3")
    axes[1].set_title(f"SINDy-RL covariance formula\ncorrelation = {corr_cov:.2f}")
    axes[1].set_xlabel("predicted std (m)")

    fig.suptitle("Predicted vs. true one-step error, real transitions", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved uncertainty_correlation.png, corr_naive={corr_naive:.3f} corr_cov={corr_cov:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trim-steps", type=int, default=3000, help="Rollout length used to find ensemble outliers.")
    parser.add_argument("--mad-k", type=float, default=5.0, help="Outlier threshold, in units of median absolute deviation.")
    parser.add_argument("--n-episode-steps", type=int, default=1000)
    parser.add_argument("--n-scatter-steps", type=int, default=5000)
    parser.add_argument("--check-idx", type=int, default=1150)
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loaded = load_swingup("data/agents/swingup/dyna", check_idx=args.check_idx)
    dyn = loaded["dyn_model"]
    dyn.set_median_coef_()
    env, policy = loaded["real_env"], loaded["policy"]
    library = dyn.model.feature_library

    coef_list = np.array(dyn.get_coef_list())
    median_coef = np.median(coef_list, axis=0)

    print("trimming ensemble outliers via a MAD rule over a real rollout...", flush=True)
    trim_states, trim_actions = collect_trim_rollout(env, policy, args.n_trim_steps)
    keep_idx = trim_ensemble(coef_list, library, trim_states, trim_actions, mad_k=args.mad_k)
    print(f"keeping {len(keep_idx)}/{coef_list.shape[0]} members: {keep_idx}")
    trimmed_coefs = coef_list[keep_idx]
    std_fn = VarianceInflation(coef_list, library, keep_idx, floor=0.0, k=1.0)

    print("collecting one episode for the tracking plot...", flush=True)
    true_ext, pred_ext, std_ext = collect_episode(env, policy, median_coef, library, std_fn, args.n_episode_steps)
    np.savez(args.out / "uncertainty_episode.npz", true_ext=true_ext, pred_ext=pred_ext, std_ext=std_ext)

    print("collecting a large batch for the correlation scatter...", flush=True)
    true_err, naive_std, cov_std = collect_scatter(env, policy, median_coef, library, std_fn, trimmed_coefs, args.n_scatter_steps)
    np.savez(args.out / "uncertainty_scatter.npz", true_err=true_err, naive_std=naive_std, cov_std=cov_std)

    plot_tracking(true_ext, pred_ext, std_ext, args.out)
    plot_correlation(true_err, naive_std, cov_std, args.out)


if __name__ == "__main__":
    main()
