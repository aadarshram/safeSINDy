"""
Dated run folder for a barrier episode.

The trajectory and the log are flushed before the video is written.
"""
import datetime as dt
from pathlib import Path

import numpy as np


def make_run_dir(parent, name, when):
    """
    Create results/<name>_<date>_<time>.

    Arguments:
        parent: directory that will hold the run folder.
        name: short name of the run.
        when: datetime of the run.

    Returns:
        Path of the new folder.
    """
    stamp = _as_datetime(when)
    folder = Path(parent) / f"{name}_{stamp:%Y-%m-%d}_{stamp:%H%M%S}"
    folder.mkdir(parents=True)
    return folder


def finalize_run(record, observations, actions, rewards, summary, frames, write_video):
    """
    Store the episode, then write the video.

    Arguments:
        record: open RunRecord.
        observations: array of shape (steps + 1, state).
        actions: array of shape (steps, action).
        rewards: array of shape (steps,).
        summary: mapping of names to values, written into the log.
        frames: rendered frames, one per step.
        write_video: callable (path, frames). It may raise.

    Returns:
        Path of the video file.
    """
    record.save_trajectory(observations, actions, rewards)
    record.write_summary(summary)
    video_path = record.run_dir / "episode.mp4"
    write_video(video_path, frames)
    return video_path


def write_episode_video(path, frames, fps=50):
    """
    Write frames to an mp4.

    Arguments:
        path: output file.
        frames: sequence of RGB arrays.
        fps: frames per second.
    """
    import imageio.v2 as imageio

    images = []
    for frame in frames:
        image = np.asarray(frame)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        images.append(image)
    imageio.mimwrite(path, images, fps=fps)
    return path


class RunRecord:
    """
    Append-only log and a trajectory file for one run folder.
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.run_dir / "log.txt", "a")

    def record_step(self, step, obs, action, reward, total):
        """
        Append one step and flush the log.
        """
        flat = np.asarray(obs, dtype=float).reshape(-1)
        force = float(np.asarray(action, dtype=float).reshape(-1)[0])
        self._log.write(
            f"step {int(step)} cart_x {flat[0]:.6f} cos {flat[1]:.6f} "
            f"force {force:.6f} reward {float(reward):.6f} return {float(total):.6f}\n"
        )
        self._log.flush()

    def save_trajectory(self, observations, actions, rewards):
        """
        Replace trajectory.npz with the arrays gathered so far.
        """
        np.savez(
            self.run_dir / "trajectory.npz",
            observations=np.asarray(observations, dtype=float),
            actions=np.asarray(actions, dtype=float),
            rewards=np.asarray(rewards, dtype=float),
        )

    def write_summary(self, summary):
        """
        Append summary lines and flush.
        """
        for key, value in summary.items():
            self._log.write(f"{key}: {value}\n")
        self._log.flush()


def _as_datetime(when):
    if isinstance(when, dt.datetime):
        return when
    text = np.datetime_as_string(np.datetime64(when), unit="s")
    return dt.datetime.fromisoformat(text)
