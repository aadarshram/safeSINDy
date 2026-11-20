"""
2D SOS sweep over (barrier position, gamma), degree-1 relaxation: checks
whether moving the barrier itself (not just gamma) ever opens a certified-
safe point for the pole constraint. It does not, at any of the 98 grid
points -- see deg2_sweep.py for the degree-2 confirmation of this same grid.

Run from inside sos/ (needs its own conda env, see sos/README.md):
    cd sos && python barrier_offset_sweep.py [--out ../results/sos/barrier_offset_sweep.csv]
"""
import argparse
import csv
import time
from pathlib import Path

from sos_sweep import SPLIT_COS2, cart_certify, pcos, pole_certify_region

X_LIMITS = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75]
GAMMAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "results" / "sos" / "barrier_offset_sweep.csv")
    args = parser.parse_args()

    rows = []
    t0 = time.time()
    for xl in X_LIMITS:
        for g in GAMMAS:
            cart_status = cart_certify(g, xl)
            a_status = pole_certify_region(g, xl, pcos**2 - SPLIT_COS2, u=1.0, h_cap=xl)
            b_status = pole_certify_region(g, xl, SPLIT_COS2 - pcos**2, u=-1.0, h_cap=xl)
            rows.append((xl, g, cart_status, a_status, b_status))
            print(f"x_limit={xl:.2f} gamma={g:5.2f}  cart={cart_status:>10}  poleA={a_status:>10}  poleB={b_status:>10}  total={time.time()-t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_limit", "gamma", "cart_status", "poleA_status", "poleB_status"])
        w.writerows(rows)
    print(f"DONE, saved {args.out}")


if __name__ == "__main__":
    main()
