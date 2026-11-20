"""
Plots the dense SOS gamma sweep (dense_sweep.py's output) as the figure used
in the presentation: a combined "fully safe" row (cart AND pole A AND pole B)
on top, broken down into the three individual constraints below it.

Usage:
    python sos/plot_dense_sweep.py [--csv ../results/sos/dense_sweep_1.5.csv] [--out ../results/figures/sos_dense_sweep.png]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent.parent / "results" / "sos" / "dense_sweep_1.5.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "results" / "figures" / "sos_dense_sweep.png")
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    gammas = np.array([float(r["gamma"]) for r in rows])

    def is_cert(s):
        return 1 if s.strip() == "optimal" else 0

    cart = np.array([is_cert(r["cart_status"]) for r in rows])
    poleA = np.array([is_cert(r["poleA_status"]) for r in rows])
    poleB = np.array([is_cert(r["poleB_status"]) for r in rows])
    combined = cart & poleA & poleB

    fig, axes = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1, 1]})
    labels = ["Fully safe\n(cart AND pole A AND pole B)", "Cart constraint", "Pole constraint\n(region A)", "Pole constraint\n(region B)"]
    series = [combined, cart, poleA, poleB]
    colors = {0: "#da3633", 1: "#2ea043"}

    for ax, lab, s in zip(axes, labels, series):
        for i in range(len(gammas) - 1):
            ax.axvspan(gammas[i], gammas[i + 1], color=colors[s[i]], alpha=0.9, lw=0)
        ax.axvspan(gammas[-1], gammas[-1] + 0.3, color=colors[s[-1]], alpha=0.9, lw=0)
        ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_xlim(gammas[0], gammas[-1])

    axes[0].set_title(f"SOS-certified safety as a function of γ — dense sweep, {len(gammas)} values, γ ∈ [{gammas[0]:.1f}, {gammas[-1]:.1f}]", fontsize=12)
    axes[-1].set_xlabel("γ")

    fig.legend(
        handles=[Patch(facecolor="#2ea043", label="SOS-certified safe"), Patch(facecolor="#da3633", label="not certified")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.text(
        0.5, 0.965, "No γ in this range is certified fully safe — the pole constraint is never certified, at any tested γ",
        ha="center", fontsize=10, style="italic", color="#444444",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
