"""
Left-right reflection, so the right-barrier HigherOrderBarrier can be reused
for a left barrier without any new barrier math.

Cart-pole dynamics are left-right symmetric: reflecting position, pole sine,
cart velocity, and pole rate (cos and position-independent terms unchanged),
while negating the applied force, maps a trajectory to its mirror image.
Wrapping the real step_fn and nominal policy in this reflection turns a
left-barrier problem into the already-validated right-barrier problem,
solved in mirrored coordinates.
"""
import numpy as np

from .barrier import HigherOrderBarrier


def reflect_state(state):
    """
    Mirror a [x, cos, sin, v, rate] state across x=0.
    """
    state = np.asarray(state, dtype=float).reshape(-1)
    return np.array([-state[0], state[1], -state[2], -state[3], -state[4]])


class ReflectedPolicy:
    """
    Wraps a controller so it sees mirrored states and returns mirrored forces.
    """

    def __init__(self, nominal):
        self.nominal = nominal

    def compute_action(self, obs):
        action = self.nominal.compute_action(reflect_state(obs))
        return -np.asarray(action, dtype=np.float32).reshape(-1)


def reflected_step_fn(step_fn):
    """
    Wraps a one-step dynamics map so it operates in mirrored coordinates.
    """

    def wrapped(state, action):
        force = -np.asarray(action, dtype=float).reshape(-1)
        nxt = step_fn(reflect_state(state), force)
        return reflect_state(nxt)

    return wrapped


class MirroredBarrier:
    """
    Wraps a HigherOrderBarrier built on reflected step_fn/nominal so callers
    can treat it exactly like a normal barrier: real obs in, real force out.
    """

    def __init__(self, inner):
        self.inner = inner

    def compute_action(self, obs):
        action = self.inner.compute_action(reflect_state(obs))
        return -np.asarray(action, dtype=np.float32).reshape(-1)

    def force_bounds(self, obs):
        """Real-coordinate feasible force interval for the mirrored barrier."""
        lo_m, hi_m = self.inner.force_bounds(reflect_state(obs))
        return -hi_m, -lo_m

    def h_min(self, obs):
        """h is a distance-to-violation; reflection doesn't change its value."""
        return self.inner.h_min(reflect_state(obs))

    def __getattr__(self, name):
        return getattr(self.inner, name)


class BilateralBarrier:
    """
    Enforces a right-side barrier and a left-side (mirrored) barrier at the
    same time: the feasible force is the intersection of both sides' feasible
    intervals, not just one side re-applied after the other.

    A one-sided barrier only corrects deviations from the nominal policy's
    preferred direction. Blocked repeatedly on one side, the system drifts
    toward the other -- including, on a rail with a real physical stop at
    both ends, into a genuine collision the one-sided filter has no way to
    see. This checks both constraints every step, not sequentially.
    """

    def __init__(self, right, left):
        self.right = right
        self.left = MirroredBarrier(left) if not isinstance(left, MirroredBarrier) else left

    def force_bounds(self, obs):
        r_lo, r_hi = self.right.force_bounds(obs)
        l_lo, l_hi = self.left.force_bounds(obs)
        lower = max(r_lo, l_lo)
        upper = min(r_hi, l_hi)
        if lower > upper + 1e-8:
            # The two sides' demands are directly contradictory at this
            # state (can happen when both are near their own panic point).
            # Defer entirely to whichever side is closer to an actual
            # violation right now (smaller h), rather than averaging or
            # picking a bound that ignores the more urgent side -- that
            # previously let the filter push toward the side demanding
            # LESS urgency while the more urgent side's own safety
            # collapsed.
            if self.right.h_min(obs) <= self.left.h_min(obs):
                return r_lo, r_hi
            return l_lo, l_hi
        return float(lower), float(upper)

    def compute_action(self, obs):
        state = np.asarray(obs, dtype=float).reshape(-1)
        force_nom = float(np.asarray(self.right.nominal.compute_action(state), dtype=float).reshape(-1)[0])
        lower, upper = self.force_bounds(state)
        clipped = float(np.clip(force_nom, lower, upper))
        return np.array([clipped], dtype=np.float32)

    def margins(self, state_k, state_k1):
        """Tighter (min) of the two sides' h and psi1 -- see HigherOrderBarrier.margins."""
        h0_r, psi1_r = self.right.margins(state_k, state_k1)
        h0_l, psi1_l = self.left.margins(state_k, state_k1)
        h0 = np.minimum(h0_r, h0_l)
        psi1 = np.minimum(psi1_r, psi1_l)
        if np.ndim(h0) == 0:
            return float(h0), float(psi1)
        return h0, psi1


def build_bilateral_barrier(
    step_fn, nominal, right_limit, left_limit,
    gamma_cart=0.5, gamma_pole=0.5, inflation=0.1, dt=0.01, u_min=-1.0, u_max=1.0,
):
    """
    The final filter design: a right-side HigherOrderBarrier and a mirrored
    left-side one, enforced together by BilateralBarrier -- what every final
    experiment (demo_comparison.py, validate_filter.py, gamma_sweep_mc.py)
    actually runs.

    Arguments:
        step_fn: one-step SINDy dynamics map, state and action in, next state out.
        nominal: the controller being filtered.
        right_limit, left_limit: world x where the right/left forbidden
            regions start (left_limit is negative; forbidden for x < left_limit).
        gamma_cart, gamma_pole: HigherOrderBarrier gains, shared by both sides.
        inflation, dt, u_min, u_max: forwarded to each HigherOrderBarrier.

    Returns:
        A BilateralBarrier with the same step_fn-based policy filtered on both sides.
    """
    right = HigherOrderBarrier(
        step_fn=step_fn, nominal=nominal, x_limit=right_limit,
        inflation=inflation, dt=dt, u_min=u_min, u_max=u_max, gamma_cart=gamma_cart, gamma_pole=gamma_pole,
    )
    left_inner = HigherOrderBarrier(
        step_fn=reflected_step_fn(step_fn), nominal=ReflectedPolicy(nominal), x_limit=-left_limit,
        inflation=inflation, dt=dt, u_min=u_min, u_max=u_max, gamma_cart=gamma_cart, gamma_pole=gamma_pole,
    )
    return BilateralBarrier(right, left_inner)
