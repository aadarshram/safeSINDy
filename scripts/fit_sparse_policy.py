"""
Fit the swing-up sparse policy and save it under data/sparse_policy.
"""
import argparse
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from safesindy.fit import fit_and_save

ROOT = Path(__file__).resolve().parents[1]


def main():
    """
    Parse paths and fit checkpoint 1150 unless another index is given.
    """
    parser = argparse.ArgumentParser(description="Fit the swing-up sparse policy and save it.")
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=ROOT / "data" / "agents" / "swingup" / "dyna",
        help="Released trial directory with params.json and checkpoints.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "sparse_policy",
        help="Directory for the npz and json.",
    )
    parser.add_argument(
        "--check-idx",
        type=int,
        default=1150,
        help="Checkpoint index. 1150 is the main swing-up network.",
    )
    args = parser.parse_args()
    json_path = fit_and_save(args.trial_dir, args.out_dir, check_idx=args.check_idx)
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
