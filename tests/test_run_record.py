import numpy as np

from safesindy.barrier_view import render_barrier_frame
from safesindy.run_record import finalize_run, make_run_dir, RunRecord


def test_run_dir_uses_the_name_and_date(tmp_path):
    when = np.datetime64("2026-11-15T19:24:00")
    run_dir = make_run_dir(tmp_path, "barrier_horizon", when)
    assert run_dir.parent == tmp_path
    assert run_dir.name == "barrier_horizon_2026-11-15_192400"
    assert run_dir.is_dir()


def test_trajectory_survives_a_video_failure(tmp_path):
    run_dir = make_run_dir(tmp_path, "barrier_horizon", np.datetime64("2026-11-15T19:24:00"))
    record = RunRecord(run_dir)
    observations = np.zeros((3, 5))
    actions = np.ones((2, 1))
    rewards = np.array([0.1, 0.2])
    record.record_step(1, observations[1], actions[0], 0.1, 0.1)

    def fail_video(path, frames):
        raise RuntimeError("encode failed")

    try:
        finalize_run(
            record,
            observations,
            actions,
            rewards,
            {"steps": 2, "return": 0.3},
            frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
            write_video=fail_video,
        )
    except RuntimeError as error:
        assert str(error) == "encode failed"
    else:
        raise AssertionError("video failure was swallowed")

    stored = np.load(run_dir / "trajectory.npz")
    assert stored["observations"].shape == (3, 5)
    assert stored["rewards"].tolist() == [0.1, 0.2]
    log = (run_dir / "log.txt").read_text()
    assert "step 1" in log
    assert "return: 0.3" in log
    assert not (run_dir / "episode.mp4").exists()


def test_front_view_barrier_fills_the_right_side():
    frame = render_barrier_frame(np.array([0.0, -1.0, 0.0, 0.0, 0.0]))
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    right_red = frame[:, -5:, 0].mean()
    left_red = frame[:, :40, 0].mean()
    assert right_red > left_red + 30
