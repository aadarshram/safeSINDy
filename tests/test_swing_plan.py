import numpy as np

from safesindy.barrier import HigherOrderBarrier
from safesindy.swing_plan import RewardHorizon

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


def _planner(lam, reward):
    barrier = HigherOrderBarrier(
        step_fn=_integrator,
        nominal=_ConstantForce(1.0),
        x_limit=0.9,
        inflation=0.1,
        dt=DT,
        gamma_cart=6.0,
        gamma_pole=6.0,
    )
    return RewardHorizon(
        step_fn=_integrator,
        nominal=_ConstantForce(1.0),
        reward_fn=reward,
        barrier=barrier,
        horizon=20,
        lam=lam,
        n_holds=2,
        grid=5,
    )


def test_zero_lambda_stays_on_the_rightward_side():
    planner = _planner(0.0, reward=lambda state, force: 0.0)
    action = planner.compute_action(np.zeros(5))
    assert float(action[0]) > 0.0


def test_swing_reward_turns_the_first_force_left():
    planner = _planner(10.0, reward=lambda state, force: -float(force))
    action = planner.compute_action(np.zeros(5))
    assert float(action[0]) < 0.0


def test_repeat_applies_the_stored_plan():
    planner = _planner(10.0, reward=lambda state, force: -float(force))
    planner.replan_every = 5
    planner.compute_action(np.zeros(5))
    planner.planned_forces[1] = 0.25
    planner.lam = 0.0
    action = planner.compute_action(np.zeros(5))
    assert abs(float(action[0]) - 0.25) < 1e-6


def test_planned_cart_stays_behind_the_inflated_limit():
    planner = _planner(0.0, reward=lambda state, force: 0.0)
    planner.compute_action(np.zeros(5))
    positions = [float(state[0]) for state in planner.predicted_states]
    assert max(positions) <= planner.barrier.safe_limit + 1e-6
