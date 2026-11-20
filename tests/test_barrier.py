import numpy as np

from safesindy.barrier import RAIL_LEFT, RAIL_RIGHT, HigherOrderBarrier, body_right, right_barrier

DT = 0.01
A_BRAKE = 10.0


def _integrator(obs, action):
    nxt = np.asarray(obs, dtype=float).copy()
    force = float(np.asarray(action, dtype=float).reshape(-1)[0])
    nxt[0] = nxt[0] + DT * nxt[3]
    nxt[3] = nxt[3] + DT * A_BRAKE * force
    return nxt


class _ConstantForce:
    def __init__(self, force):
        self.force = float(force)

    def compute_action(self, obs):
        return np.array([self.force], dtype=np.float32)


def _shield(force, inflation, gamma_cart=6.0, gamma_pole=6.0):
    return HigherOrderBarrier(
        step_fn=_integrator,
        nominal=_ConstantForce(force),
        x_limit=0.9,
        inflation=inflation,
        dt=DT,
        gamma_cart=gamma_cart,
        gamma_pole=gamma_pole,
    )


def test_right_barrier_leaves_room_for_a_horizontal_pole():
    limit = right_barrier(RAIL_LEFT, RAIL_RIGHT)
    assert np.isclose(limit, 1.5)


def test_body_right_uses_the_cart_or_the_pole_tip():
    upright = body_right(0.2, np.sin(0.0))
    toward_barrier = body_right(0.2, np.sin(np.pi / 2))
    assert upright == np.float64(0.2 + 0.2)
    assert toward_barrier > upright
    assert toward_barrier == np.float64(0.2 + 1.0 + 0.045)


def test_slow_cart_keeps_the_nominal_force():
    shield = _shield(0.2, inflation=0.0)
    action = shield.compute_action(np.array([-1.0, -1.0, 0.0, 0.0, 0.0]))
    assert abs(float(action[0]) - 0.2) < 1e-6


def test_fast_cart_brakes_while_still_short_of_the_barrier():
    shield = _shield(1.0, inflation=0.0)
    state = np.array([0.3, 1.0, 0.0, 2.3, 0.0])
    action = shield.compute_action(state)
    assert state[0] < shield.x_limit
    assert float(action[0]) < 1.0
    nxt = _integrator(state, action)
    _, slack = shield.margins(state, nxt)
    assert float(slack) >= -1e-6


def test_clipped_force_satisfies_the_psi1_decay_bound():
    shield = _shield(1.0, inflation=0.1)
    state = np.array([0.2, 1.0, 0.0, 0.4, 0.0])
    force = float(shield.clip_force(state, 1.0))
    nxt = _integrator(state, np.array([force]))
    h0, slack = shield.margins(state, nxt)
    assert h0 > 0.0
    assert float(slack) >= -1e-6


def test_horizontal_pole_brakes_while_the_cart_edge_is_still_clear():
    shield = HigherOrderBarrier(
        step_fn=_integrator,
        nominal=_ConstantForce(1.0),
        x_limit=1.5,
        inflation=0.0,
        dt=DT,
    )
    state = np.array([0.3, 0.0, 1.0, 0.0, 0.0])
    action = shield.compute_action(state)
    assert body_right(state[0], state[2]) < shield.x_limit
    assert float(action[0]) < 1.0
    nxt = _integrator(state, action)
    assert body_right(nxt[0], nxt[2]) < shield.x_limit


def test_inflated_limit_holds_on_the_double_integrator():
    inflation = 0.1
    shield = _shield(1.0, inflation=inflation)
    assert shield.safe_limit == 0.9 - inflation
    state = np.zeros(5)
    positions = []
    for _ in range(500):
        action = shield.compute_action(state)
        state = _integrator(state, action)
        positions.append(state[0])
    extent = [body_right(x, 0.0) for x in positions]
    assert max(extent) <= shield.safe_limit + 1e-6
    assert max(positions) > 0.4
