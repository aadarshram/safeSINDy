"""
Roll one episode of the network and one of the saved sparse policy.
"""
import argparse
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import imageio.v2 as imageio
import numpy as np

from safesindy.load import load_swingup
from safesindy.report import write_report
from safesindy.rollout import rollout_episode
from safesindy.sparse_policy import load_sparse_policy

ROOT = Path(__file__).resolve().parents[1]
VIDEO_FPS = 50


def main():
    """
    Load both controllers, roll one shared-seed episode each, and write the report.
    """
    parser = argparse.ArgumentParser(description="Roll one swing-up episode of each controller.")
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=ROOT / "data" / "agents" / "swingup" / "dyna",
        help="Released trial directory with params.json and checkpoints.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "data" / "sparse_policy" / "cart_swingup_1150.json",
        help="Json sidecar of the saved sparse policy.",
    )
    parser.add_argument(
        "--check-idx",
        type=int,
        default=1150,
        help="Checkpoint index of the neural policy.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shared environment reset seed.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for the text report and the two videos.",
    )
    args = parser.parse_args()

    sparse = load_sparse_policy(args.policy)
    loaded = load_swingup(args.trial_dir, check_idx=args.check_idx)
    real_env = loaded["real_env"]

    nn_episode = rollout_episode(real_env, loaded["policy"], seed=args.seed)
    sindy_episode = rollout_episode(real_env, sparse, seed=args.seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    videos = {
        "nn": args.results_dir / "nn_episode.mp4",
        "sindy": args.results_dir / "sindy_episode.mp4",
    }
    _write_video(videos["nn"], nn_episode.frames)
    _write_video(videos["sindy"], sindy_episode.frames)
    report_path = args.results_dir / "one_episode.txt"
    write_report(
        report_path,
        seed=args.seed,
        checkpoint=args.check_idx,
        expression=sparse.expression,
        episodes={"nn": nn_episode, "sindy": sindy_episode},
        videos=videos,
    )
    print(report_path.read_text())


def _write_video(path, frames):
    images = []
    for frame in frames:
        image = np.asarray(frame)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        images.append(image)
    imageio.mimwrite(path, images, fps=VIDEO_FPS)


if __name__ == "__main__":
    main()
