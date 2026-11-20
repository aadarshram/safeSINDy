"""
Two-region SOS certificate for the pole-tip barrier.

Region A (|cos| close to 1, pole near upright or near hanging straight
down): u = u_max helps psi1_pole most.
Region B (|cos| small, pole near horizontal): u = u_min helps most.
Split point cos^2 ~= 0.6863, derived by hand from d(h_dot_next)/du's sign
change (see conversation notes). Each region is certified independently.
"""
import itertools
import sympy as sp
from manual_sos import certify

# fitted SINDy coefficients (median model, checkpoint 1150)
a, b = 0.999980, 0.009990          # x_next = a*x + b*v
c_cos, d_cos = 0.999960, -0.009990  # cos_next = c_cos*cos + d_cos*sin*rate
c_sin, d_sin = 0.999181, 0.009963   # sin_next = c_sin*sin + d_sin*cos*rate + u*p2sin(cos)
p2sin_const, p2sin_cos2, p2sin_sin2 = 0.388925, -0.389603, -0.388889
c_v, = (0.997648,)                  # v_next = c_v*v + u*p2v(cos)
p2v_const, p2v_cos2, p2v_sin2 = 0.067839, 0.026940, 0.024254
e_rate_sin, c_rate = 0.147576, 1.000094  # rate_next = e_rate_sin*sin + c_rate*rate + u*p2rate(cos)
p2rate_cos = -0.136857

POLE_RADIUS, POLE_LENGTH = 0.045, 1.0
CART_HALF = 0.2
SPLIT_COS2 = 0.6863

x, cos, sin, v, rate = sp.symbols("x cos sin v rate")

BOX = [
    (x + 1.8) * (1.8 - x),
    (1 - cos**2),
    (1 - sin**2),
    (v + 4.0) * (4.0 - v),
    (rate + 3.5) * (3.5 - rate),
]


def h_pole(safe_limit, xv, sinv):
    return safe_limit - POLE_RADIUS - xv - POLE_LENGTH * sinv


def h_dot_pole(vv, cosv, ratev):
    return -vv - POLE_LENGTH * cosv * ratev


def next_state(u):
    x_next = a * x + b * v
    cos_next = c_cos * cos + d_cos * sin * rate
    p2sin = p2sin_const + p2sin_cos2 * cos**2 + p2sin_sin2 * sin**2
    sin_next = c_sin * sin + d_sin * cos * rate + u * p2sin
    p2v = p2v_const + p2v_cos2 * cos**2 + p2v_sin2 * sin**2
    v_next = c_v * v + u * p2v
    p2rate = p2rate_cos * cos
    rate_next = e_rate_sin * sin + c_rate * rate + u * p2rate
    return x_next, cos_next, sin_next, v_next, rate_next


def build(gamma, x_limit, u):
    safe_limit = x_limit
    psi1 = gamma * h_pole(safe_limit, x, sin) + h_dot_pole(v, cos, rate)
    xn, cosn, sinn, vn, raten = next_state(u)
    psi1_next = gamma * h_pole(safe_limit, xn, sinn) + h_dot_pole(vn, cosn, raten)
    return sp.expand(psi1_next), psi1


CIRCLE = sp.expand(cos**2 + sin**2 - 1)


def certify_region(gamma, x_limit, region_ineq, u, deg, ineq_prods=False, solver=None, h_cap=None):
    """
    h_cap restricts certification to the "danger zone" h_pole(x) <= h_cap,
    rather than the whole rail. Far from the wall (h huge), a one-step
    psi1 check can fail for reasons that have nothing to do with real risk
    (gamma*h barely matters there), so certifying the full rail is strictly
    more than what safety actually requires.
    """
    psi1_next, psi1 = build(gamma, x_limit, u)
    ineqs = BOX + [sp.expand(psi1), sp.expand(region_ineq)]
    if h_cap is not None:
        ineqs.append(sp.expand(h_cap - h_pole(x_limit, x, sin)))
    kwargs = {"ineq_prods": ineq_prods, "eqs": [CIRCLE]}
    if solver is not None:
        kwargs["solver"] = solver
    status, prob = certify([x, cos, sin, v, rate], psi1_next, ineqs, deg=deg, **kwargs)
    return status, prob


def certify_pole(gamma, x_limit, deg=1, ineq_prods=False):
    """Certify both regions; returns (status_A, status_B)."""
    status_a, _ = certify_region(gamma, x_limit, cos**2 - SPLIT_COS2, u=1.0, deg=deg, ineq_prods=ineq_prods)
    status_b, _ = certify_region(gamma, x_limit, SPLIT_COS2 - cos**2, u=-1.0, deg=deg, ineq_prods=ineq_prods)
    return status_a, status_b


if __name__ == "__main__":
    import sys
    gamma = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    x_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    sa, sb = certify_pole(gamma, x_limit, deg=deg)
    print(f"gamma={gamma} x_limit={x_limit} deg={deg} regionA={sa} regionB={sb}")
