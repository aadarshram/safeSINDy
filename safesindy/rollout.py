"""
One-episode rollout with one rendered frame per step.
"""
from dataclasses import dataclass

import numpy as np

FRAME_HEIGHT = 480
FRAME_WIDTH = 480
STEP_GUARD = 2000


@dataclass
class Episode:
    """
    Rewards, frames, observations, and actions from one episode.
    """

    rewards: np.ndarray
    frames: list
    observations: np.ndarray
    actions: np.ndarray

    @property
    def reward_sum(self):
        return float(np.sum(self.rewards))

    @property
    def n_steps(self):
        return int(len(self.rewards))


def rollout_episode(env, policy, seed, max_steps=STEP_GUARD):
    """
    Reset with seed and step until the episode ends.

    Arguments:
        env: environment with reset, step, and render.
        policy: object with compute_action.
        seed: reset seed.
        max_steps: stop with an error if the episode has not ended.

    Returns:
        Episode. The number of frames equals the number of steps.
    """
    obs = _obs_from_reset(env.reset(seed=seed))
    rewards = []
    frames = []
    observations = [np.asarray(obs, dtype=float)]
    actions = []
    for _ in range(max_steps):
        action = np.asarray(policy.compute_action(obs), dtype=np.float32).reshape(-1)
        obs, reward, done, _info = _unpack_step(env.step(action))
        rewards.append(float(reward))
        actions.append(action.copy())
        frames.append(_grab_frame(env))
        observations.append(np.asarray(obs, dtype=float))
        if done:
            break
    else:
        raise RuntimeError(f"episode did not end within {max_steps} steps")
    return Episode(
        rewards=np.asarray(rewards, dtype=float),
        frames=frames,
        observations=np.asarray(observations),
        actions=np.asarray(actions),
    )


def _obs_from_reset(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def _unpack_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


def _grab_frame(env):
    if hasattr(env, "_height"):
        env._height = FRAME_HEIGHT
        env._width = FRAME_WIDTH
    if hasattr(env, "render"):
        try:
            frame = env.render()
        except (NotImplementedError, TypeError):
            frame = None
        if frame is not None:
            return np.asarray(frame)
    inner = getattr(env, "_env", None)
    if inner is not None and hasattr(inner, "physics"):
        return np.asarray(inner.physics.render(height=FRAME_HEIGHT, width=FRAME_WIDTH))
    raise RuntimeError("environment has no render")
