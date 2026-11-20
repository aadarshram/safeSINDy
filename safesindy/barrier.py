"""
Discrete-time barrier on the rightmost point of the cart and the pole.

The shaded world region starts at 1.5 m. That is shorter than the old 0.9 m
slab, so a pole laid toward the barrier still has room to swing. The nominal
force is the released network. Each constraint is checked two SINDy steps
ahead on the real one-step map (no derivative, no continuous-time formula
discretized after the fact). Both constraints use an inflated limit inside
the shade, so a small model error is not an impossible stop.
"""
import numpy as np

RAIL_LEFT = -1.8
RAIL_RIGHT = 1.8
RIGHT_FRACTION = 11.0 / 12.0
CART_HALF = 0.2
POLE_LENGTH = 1.0
POLE_RADIUS = 0.045


def right_barrier(x_left=RAIL_LEFT, x_right=RAIL_RIGHT, fraction=RIGHT_FRACTION):
    """
    World position where the forbidden right-hand interval starts.

    Arguments:
        x_left: left end of the rail.
        x_right: right end of the rail.
        fraction: fraction of the rail, measured from the left end.

    Returns:
        World x, in meters. A cart or pole point further right is forbidden.
    """
    return float(x_left + fraction * (x_right - x_left))


def body_right(position, sin_theta):
    """
    Rightmost point of the cart box and the pole capsule.

    Arguments:
        position: cart center, in meters.
        sin_theta: sine of the pole angle. Positive points toward the barrier.

    Returns:
        World x of the rightmost point, in meters.
    """
    pole_reach = np.maximum(0.0, np.asarray(sin_theta, dtype=float)) * POLE_LENGTH + POLE_RADIUS
    overhang = np.maximum(CART_HALF, pole_reach)
    return np.asarray(position, dtype=float) + overhang


def left_barrier(x_left=RAIL_LEFT, x_right=RAIL_RIGHT, fraction=RIGHT_FRACTION):
    """
    World position where the forbidden left-hand interval starts. Mirror of
    right_barrier: a cart or pole point further left than this is forbidden.
    """
    return float(x_right - fraction * (x_right - x_left))


def body_left(position, sin_theta):
    """
    Leftmost point of the cart box and the pole capsule. Mirror of body_right.
    """
    pole_reach = np.maximum(0.0, -np.asarray(sin_theta, dtype=float)) * POLE_LENGTH + POLE_RADIUS
    overhang = np.maximum(CART_HALF, pole_reach)
    return np.asarray(position, dtype=float) - overhang


def _as_state(obs):
    return np.asarray(obs, dtype=float).reshape(-1)


def _as_force(action):
    return float(np.asarray(action, dtype=float).reshape(-1)[0])


class HigherOrderBarrier:
    """
    Discrete-time higher-order barrier on the cart edge and the pole tip.

    h_cart = safe_limit - cart_half - x.
    h_pole = safe_limit - pole_radius - x - pole_length sin(theta).

    Position does not move with u_k within one SINDy step -- only velocity
    does, position catches up the step after (explicit-Euler position
    update, confirmed empirically: x_{k+1} = x_k + dt*v_k to within the
    model's own ~1e-5 fit residual). So h itself is not one-step-controllable,
    but psi1 = gamma*h - h_dot is, because h_dot is exactly linear in
    velocity, and velocity reacts to u_k in one real SINDy step.

    Soundness only needs ONE real model step: if psi1(x_k) >= 0 and we
    enforce psi1(x_{k+1}) >= 0 using the real one-step prediction (not a
    repeat-the-same-force assumption -- the same check reruns fresh next
    step regardless of what u_{k+1} turns out to be), then psi1(x_k) >= 0
    for all k by induction, which forces h(x_{k+1}) = h(x_k) - dt*h_dot(x_k)
    >= h(x_k)*(1 - dt*gamma) >= 0 whenever h(x_k) >= 0 and gamma < 1/dt.
    No derivative-of-a-derivative, no continuous-time formula discretized
    after the fact: gamma > 0 is the bandwidth knob (weighting h against
    h_dot inside psi1), not restricted to (0, 1] -- that range belongs to
    a decay *fraction*, which this does not use: psi1 must simply stay
    non-negative, period.

    Cart and pole get separate gammas. h_dot_cart = -v is bounded only by
    the rail's own velocity range, but h_dot_pole = -v - L*rate*cos(theta)
    scales with the pole's swing rate, which is large during any ordinary
    swing-up or balance correction -- nowhere near the barrier. A single
    shared gamma that is large enough to brake decisively near the wall is
    also too small relative to that ordinary swing rate far from it, so it
    ends up treating normal swinging as a standing violation. Splitting
    the two lets gamma_pole be calibrated against the pole's own velocity
    range instead of reusing whatever the cart needs.
    """

    def __init__(
        self,
        step_fn,
        nominal,
        x_limit,
        inflation=0.1,
        dt=0.01,
        u_min=-1.0,
        u_max=1.0,
        gamma_cart=0.3,
        gamma_pole=0.3,
    ):
        self.step_fn = step_fn
        self.nominal = nominal
        self.x_limit = float(x_limit)
        self.inflation = float(inflation)
        self.safe_limit = self.x_limit - self.inflation
        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        if gamma_cart <= 0.0 or gamma_pole <= 0.0:
            raise ValueError("gamma_cart and gamma_pole must be positive")
        self.gamma_cart = float(gamma_cart)
        self.gamma_pole = float(gamma_pole)

    def compute_action(self, obs):
        """
        Project the nominal force onto both barriers.

        Arguments:
            obs: current observation. Entries are x, cos, sin, dx, dtheta.

        Returns:
            Force as float32, shape (1,).
        """
        state = _as_state(obs)
        force_nom = float(np.clip(_as_force(self.nominal.compute_action(state)), self.u_min, self.u_max))
        return np.array([self.clip_force(state, force_nom)], dtype=np.float32)

    def clip_force(self, obs, force):
        """
        Force closest to the request that meets both barriers.

        Arguments:
            obs: current observation.
            force: requested force.

        Returns:
            Clipped force as a float.
        """
        state = _as_state(obs)
        force_nom = float(np.clip(force, self.u_min, self.u_max))
        lower, upper = self.force_bounds(state)
        return float(np.clip(force_nom, lower, upper))

    def force_bounds(self, obs):
        """
        Forces that keep both barriers' one-step psi1 decay bound satisfiable.

        Arguments:
            obs: current observation.

        Returns:
            Lower and upper force. An infeasible state returns the brake at both ends.
        """
        state = _as_state(obs)
        lo_cart, lo_pole = self._psi1_next(state, self.u_min)
        hi_cart, hi_pole = self._psi1_next(state, self.u_max)

        lower, upper = self.u_min, self.u_max
        for lo, hi in ((lo_cart, hi_cart), (lo_pole, hi_pole)):
            bounds = self._feasible_u(lo, hi, 0.0)
            if bounds is None:
                return self.u_min, self.u_min
            lower = max(lower, bounds[0])
            upper = min(upper, bounds[1])
        if lower > upper + 1e-8:
            return self.u_min, self.u_min
        return float(lower), float(upper)

    def h_min(self, obs):
        """
        Tighter of the two barriers' current safety margin: how close this
        state already is to a violation, independent of gamma or force.
        """
        state = _as_state(obs)
        return min(self._h_cart(state), self._h_pole(state))

    def _h_cart(self, state):
        return self.safe_limit - CART_HALF - float(state[0])

    def _h_pole(self, state):
        return self.safe_limit - POLE_RADIUS - float(state[0]) - POLE_LENGTH * float(state[2])

    def _psi1_cart(self, state):
        return self.gamma_cart * self._h_cart(state) - float(state[3])

    def _psi1_pole(self, state):
        h_dot = -float(state[3]) - POLE_LENGTH * float(state[1]) * float(state[4])
        return self.gamma_pole * self._h_pole(state) + h_dot

    def _psi1_next(self, state, force):
        """
        psi1 one real SINDy step ahead, for a candidate force held this step only.
        """
        nxt = _as_state(self.step_fn(state, np.array([force], dtype=float)))
        return self._psi1_cart(nxt), self._psi1_pole(nxt)

    def _feasible_u(self, h_lo, h_hi, required):
        """
        Sub-interval of [u_min, u_max] where h(u) >= required.

        h is assumed monotone between h_lo (at u_min) and h_hi (at u_max),
        which is what the real two-step SINDy map gives for these barriers.
        """
        span = self.u_max - self.u_min
        slope = (h_hi - h_lo) / span
        if abs(slope) <= 1e-9:
            if h_lo >= required - 1e-9:
                return self.u_min, self.u_max
            return None
        boundary = self.u_min + (required - h_lo) / slope
        if slope > 0.0:
            lower = max(self.u_min, boundary)
            if lower > self.u_max + 1e-9:
                return None
            return lower, self.u_max
        upper = min(self.u_max, boundary)
        if upper < self.u_min - 1e-9:
            return None
        return self.u_min, upper

    def margins(self, state_k, state_k1):
        """
        h and psi1 one step later, for a real trajectory.

        Arguments:
            state_k, state_k1: two consecutive real states (x_k, x_{k+1}),
                each shape (5,) or a batch (n, 5).

        Returns:
            h0, psi1_next. h0 is h(x_k) (the tighter of cart/pole). psi1_next
            is psi1(x_{k+1}) (the tighter of cart/pole) -- non-negative means
            the bound held on this real transition. Scalars stay scalars.
        """
        single = np.asarray(state_k).ndim == 1
        state_k = np.atleast_2d(np.asarray(state_k, dtype=float))
        state_k1 = np.atleast_2d(np.asarray(state_k1, dtype=float))

        h0_cart = self.safe_limit - CART_HALF - state_k[:, 0]
        h0_pole = self.safe_limit - POLE_RADIUS - state_k[:, 0] - POLE_LENGTH * state_k[:, 2]

        h1_cart = self.safe_limit - CART_HALF - state_k1[:, 0]
        h1_pole = self.safe_limit - POLE_RADIUS - state_k1[:, 0] - POLE_LENGTH * state_k1[:, 2]
        psi1_cart1 = self.gamma_cart * h1_cart - state_k1[:, 3]
        h_dot_pole1 = -state_k1[:, 3] - POLE_LENGTH * state_k1[:, 1] * state_k1[:, 4]
        psi1_pole1 = self.gamma_pole * h1_pole + h_dot_pole1

        h0 = np.minimum(h0_cart, h0_pole)
        psi1_next = np.minimum(psi1_cart1, psi1_pole1)
        if single:
            return float(h0[0]), float(psi1_next[0])
        return h0, psi1_next
