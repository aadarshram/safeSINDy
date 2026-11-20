import numpy as np

from safesindy.trace_plot import axis_limits, frames_with_trace, psi_limits, trace_image


def test_axes_are_fixed_across_runs():
    reward_limits, barrier_limits, step_limits = axis_limits(
        np.array([0.1, 0.4, 0.2]),
        np.array([0.9, 0.3, 0.5]),
    )
    assert reward_limits == (0.0, 1.0)
    assert barrier_limits == (-1.6, 3.4)
    assert step_limits == (0, 999)


def test_trace_grows_with_the_frame():
    rewards = np.linspace(0.0, 1.0, 20)
    barriers = np.linspace(0.9, 0.2, 20)
    early = trace_image(rewards, barriers, index=1, width=640, height=180)
    late = trace_image(rewards, barriers, index=19, width=640, height=180)
    assert early.shape == (180, 640, 3)
    assert early.dtype == np.uint8
    assert not np.array_equal(early, late)


def test_psi_axes_are_fixed():
    assert psi_limits(np.array([0.2, -0.1]), np.array([1.0, -3.0])) == ((-8.0, 12.0), (-40.0, 60.0))


def test_psi_panel_is_attached_under_the_safety_trace():
    frame = np.zeros((4, 640, 3), dtype=np.uint8)
    rewards = np.array([0.0, 0.5])
    barriers = np.array([0.8, 0.4])
    psi1 = np.array([1.0, 0.2])
    psi2 = np.array([4.0, -1.0])
    plain = frames_with_trace([frame, frame], rewards, barriers, plot_height=40)
    stacked = frames_with_trace(
        [frame, frame],
        rewards,
        barriers,
        plot_height=40,
        psi1=psi1,
        psi2=psi2,
    )
    assert stacked[0].shape[0] == plain[0].shape[0] + 40
    other = psi1 + 3.0
    changed = frames_with_trace(
        [frame, frame],
        rewards,
        barriers,
        plot_height=40,
        psi1=other,
        psi2=psi2,
    )
    assert not np.array_equal(stacked[1], changed[1])


def test_trace_is_attached_under_the_cart_frame():
    frame = np.zeros((4, 640, 3), dtype=np.uint8)
    rewards = np.array([0.0, 0.5])
    barriers = np.array([0.8, 0.4])
    stacked = frames_with_trace([frame, frame], rewards, barriers, plot_height=40)
    assert len(stacked) == 2
    assert stacked[0].shape[0] > frame.shape[0]
    assert stacked[0].shape[1] == 640
    assert not np.array_equal(stacked[0], stacked[1])
