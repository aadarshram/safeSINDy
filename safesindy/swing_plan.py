"""
Receding-horizon swing program on the one-step model.

The cost is the squared gap from the nominal force, minus lambda times the
swing reward. The higher-order barrier clips every force on the rollout.
"""
import itertools

import numpy as np

from safesindy.barrier import body_right


def _as_state(obs):
    return np.asarray(obs, dtype=float).reshape(-1)


def _as_force(action):
    return float(np.asarray(action, dtype=float).reshape(-1)[0])


class RewardHorizon:
    """
    Grid search over held forces for one horizon of the swing program.
    """

    def __init__(self, step_fn, nominal, reward_fn, barrier, horizon, lam, n_holds=2, grid=5, replan_every=1):
        self.horizon = int(horizon)
        self.n_holds = int(n_holds)
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.n_holds < 1 or self.horizon % self.n_holds != 0:
            raise ValueError("n_holds must divide horizon")
        self.grid = int(grid)
        if self.grid < 2:
            raise ValueError("grid must have at least 2 levels")
        self.step_fn = step_fn
        self.nominal = nominal
        self.reward_fn = reward_fn
        self.barrier = barrier
        self.lam = float(lam)
        self.replan_every = int(replan_every)
        if self.replan_every < 1:
            raise ValueError("replan_every must be at least 1")
        self.predicted_states = None
        self.planned_forces = None
        self._age = 0

    def compute_action(self, obs):
        """
        Search the held-force grid and return the first applied force.

        Arguments:
            obs: current observation.

        Returns:
            Force as float32, shape (1,).

        A fresh search runs on the first call and again every replan_every calls.
        Between searches the stored plan is applied in order.
        """
        if self.planned_forces is not None and self._age < self.replan_every:
            index = min(self._age, len(self.planned_forces) - 1)
            force = self.planned_forces[index]
            self._age += 1
            return np.array([force], dtype=np.float32)
        x0 = _as_state(obs)
        levels = np.linspace(self.barrier.u_min, self.barrier.u_max, self.grid)
        best_cost = np.inf
        best_forces = None
        best_states = None
        for knots in itertools.product(levels, repeat=self.n_holds):
            cost, forces, states = self._evaluate(x0, knots)
            if cost < best_cost:
                best_cost = cost
                best_forces = forces
                best_states = states
        if best_forces is None:
            best_forces = np.full(self.horizon, self.barrier.u_min)
            best_states = [x0]
        self.planned_forces = np.asarray(best_forces, dtype=float)
        self.predicted_states = best_states
        self._age = 1
        return np.array([self.planned_forces[0]], dtype=np.float32)

    def _evaluate(self, x0, knots):
        hold = self.horizon // self.n_holds
        state = np.asarray(x0, dtype=float)
        states = [state.copy()]
        forces = []
        tracking = 0.0
        reward = 0.0
        try:
            for index in range(self.horizon):
                requested = float(knots[index // hold])
                reference = float(
                    np.clip(
                        _as_force(self.nominal.compute_action(state)),
                        self.barrier.u_min,
                        self.barrier.u_max,
                    )
                )
                force = float(self.barrier.clip_force(state, requested))
                tracking += (force - reference) ** 2
                reward += float(self.reward_fn(state, force))
                nxt = _as_state(self.step_fn(state, np.array([force], dtype=float)))
                forces.append(force)
                states.append(nxt)
                state = nxt
                if float(body_right(state[0], state[2])) > self.barrier.safe_limit + 1e-6:
                    return np.inf, forces, states
        except ValueError:
            return np.inf, forces, states
        return tracking - self.lam * reward, forces, states
