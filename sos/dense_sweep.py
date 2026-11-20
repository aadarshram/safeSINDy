"""
Dense 1D SOS gamma sweep at the default barrier position (x_limit=1.5): the
centerpiece negative result -- no gamma in a fine 52-point grid is SOS-
certified safe for the pole constraint, while the cart constraint certifies
safe up to about gamma=2.0. Plotted by plot_dense_sweep.py.

Run from inside sos/ (needs its own conda env, see sos/README.md):
    cd sos && python dense_sweep.py [--out ../results/sos/dense_sweep_1.5.csv]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sos_sweep import SPLIT_COS2, cart_certify, pcos, pole_certify_region

X_LIMIT = 1.5
GAMMAS = sorted(set(
    list(np.round(np.arange(0.1, 3.01, 0.1), 2))
    + list(np.round(np.arange(3.0, 10.01, 0.5), 2))
    + list(np.round(np.arange(1.8, 2.61, 0.05), 2))
))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "results" / "sos" / "dense_sweep_1.5.csv")
    args = parser.parse_args()

    rows = []
    t_start = time.time()
    for gamma in GAMMAS:
        t0 = time.time()
        cart_status = cart_certify(float(gamma), X_LIMIT)
        a_status = pole_certify_region(float(gamma), X_LIMIT, pcos**2 - SPLIT_COS2, u=1.0)
        b_status = pole_certify_region(float(gamma), X_LIMIT, SPLIT_COS2 - pcos**2, u=-1.0)
        rows.append((gamma, cart_status, a_status, b_status))
        print(f"{gamma:6.2f}  {cart_status:>12}  {a_status:>12}  {b_status:>12}   ({time.time()-t0:.1f}s)  total={time.time()-t_start:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "cart_status", "poleA_status", "poleB_status"])
        w.writerows(rows)
    print(f"DONE, saved {args.out}")


if __name__ == "__main__":
    main()
