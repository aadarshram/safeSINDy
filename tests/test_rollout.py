import numpy as np

from safesindy.report import write_report
from safesindy.rollout import rollout_episode


class _HoldPolicy:
    """
    Policy that returns a constant action.
    """

    def compute_action(self, obs):
        return np.array([0.25], dtype=np.float32)


class _FakeEnv:
    """
    Environment that pays a prescribed reward and then ends.
    """

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self._index = 0
        self._steps = 0

    def reset(self, seed=None):
        self._index = 0
        self._steps = 0
        return np.zeros(5), {}

    def step(self, action):
        reward = self._rewards[self._index]
        self._index += 1
        self._steps += 1
        done = self._index >= len(self._rewards)
        obs = np.full(5, self._steps, dtype=float)
        return obs, reward, done, False, {}

    def render(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[..., 0] = self._steps
        return frame


def test_rollout_sums_rewards_and_keeps_one_frame_per_step():
    """
    Three rewards sum, and each step stores one frame.
    """
    episode = rollout_episode(_FakeEnv([1.0, 2.0, 3.0]), _HoldPolicy(), seed=0)
    assert episode.reward_sum == 6.0
    assert len(episode.frames) == 3
    assert episode.n_steps == 3


def test_immediate_done_writes_report(tmp_path):
    """
    An episode that ends on the first step still writes a report.
    """
    episode = rollout_episode(_FakeEnv([4.5]), _HoldPolicy(), seed=1)
    report_path = tmp_path / "one_episode.txt"
    write_report(
        report_path,
        seed=1,
        checkpoint=1150,
        expression="u0 = 0.500 x",
        episodes={"nn": episode, "sindy": episode},
        videos={
            "nn": tmp_path / "nn_episode.mp4",
            "sindy": tmp_path / "sindy_episode.mp4",
        },
    )
    text = report_path.read_text()
    assert "4.5" in text
    assert "1150" in text
    assert "u0 = 0.500 x" in text
