"""
SOS certificate as the primary verification method: sweep gamma, run the
actual Gram-matrix certificate (not Monte Carlo) at each value, for both
the cart and the two pole regions. deg=1 is fast (~1-5s per point) and
reliable away from thin margins -- see sos/README.md for that limitation.
"""
import sys
import time
import sympy as sp

from manual_sos import certify
from pole_cert import build as pole_build, x as px, cos as pcos, sin as psin, v as pv, rate as prate, SPLIT_COS2, h_pole

a, b, c = 0.999980, 0.009990, 0.997648
p2v_const, p2v_cos2, p2v_sin2 = 0.067839, 0.026940, 0.024254
CART_HALF = 0.2
cx, cv, ccos = sp.symbols("x v cos")


def cart_certify(gamma, x_limit, deg=1):
    h = x_limit - CART_HALF - cx
    psi1 = gamma * h - cv
    x_next = a * cx + b * cv
    p2v = p2v_const + p2v_cos2 * ccos**2 + p2v_sin2 * (1 - ccos**2)
    v_next = c * cv - p2v
    h_next = x_limit - CART_HALF - x_next
    psi1_next = gamma * h_next - v_next
    box = [(cx + 1.8) * (1.8 - cx), (cv + 5.0) * (5.0 - cv), (1 - ccos**2)]
    status, _ = certify([cx, cv, ccos], psi1_next, box + [psi1], deg=deg)
    return status


def pole_certify_region(gamma, x_limit, region_ineq, u, deg=1, h_cap=1.5):
    psi1_next, psi1 = pole_build(gamma, x_limit, u)
    ineqs = [
        (px + 1.8) * (1.8 - px),
        (pv + 4.0) * (4.0 - pv),
        (prate + 3.5) * (3.5 - prate),
        sp.expand(psi1),
        sp.expand(region_ineq),
        sp.expand(h_cap - h_pole(x_limit, px, psin)),
    ]
    circle = sp.expand(pcos**2 + psin**2 - 1)
    status, _ = certify([px, pcos, psin, pv, prate], psi1_next, ineqs, deg=deg, eqs=[circle])
    return status


def main():
    x_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    gammas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
    print(f"x_limit={x_limit}")
    print(f"{'gamma':>6}  {'cart':>12}  {'poleA':>12}  {'poleB':>12}")
    for gamma in gammas:
        t0 = time.time()
        cart_status = cart_certify(gamma, x_limit)
        a_status = pole_certify_region(gamma, x_limit, pcos**2 - SPLIT_COS2, u=1.0)
        b_status = pole_certify_region(gamma, x_limit, SPLIT_COS2 - pcos**2, u=-1.0)
        print(f"{gamma:6.2f}  {cart_status:>12}  {a_status:>12}  {b_status:>12}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
