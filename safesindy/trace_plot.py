"""
Reward, safety margin, and higher-order barrier traces under each video frame.

The scales are fixed so a filtered run and an unfiltered run share one chart.
Reward runs from 0 to 1. The safety axis is the clearance of the cart and the
pole to the barrier, in meters, from -1.6 to 3.4. Above 0 the whole body is
left of the shade.

The lower chart is the tighter of the cart-edge and pole-tip constraints.
psi1 uses the left axis and psi2 the right. The two zero lines lie on one
horizontal line. The ends cover a cart on the rail at up to 6 m/s and 12 m/s^2,
with gamma from a 9.5 m/s^2 brake.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REWARD_LIMITS = (0.0, 1.0)
BARRIER_LIMITS = (-1.6, 3.4)
STEP_LIMITS = (0, 999)
PSI1_LIMITS = (-8.0, 12.0)
PSI2_LIMITS = (-40.0, 60.0)


def axis_limits(rewards, barriers):
    """
    Fixed axis ends, shared by every swing-up episode.

    Arguments:
        rewards: unused. The reward axis is always 0 to 1.
        barriers: unused. The safety axis is always -1.6 to 3.4.

    Returns:
        (reward_min, reward_max), (barrier_min, barrier_max), (step_min, step_max).
    """
    del rewards, barriers
    return REWARD_LIMITS, BARRIER_LIMITS, STEP_LIMITS


def psi_limits(psi1, psi2):
    """
    Fixed axis ends for the two higher-order constraints.

    Arguments:
        psi1: unused. The left axis is always -8 to 12.
        psi2: unused. The right axis is always -40 to 60, so its zero
            sits on the same horizontal line as the psi1 zero.

    Returns:
        (psi1_min, psi1_max), (psi2_min, psi2_max).
    """
    del psi1, psi2
    return PSI1_LIMITS, PSI2_LIMITS


def _draw(width, height):
    figure, left = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    return figure, left


def _rgb(figure, width, height):
    figure.subplots_adjust(left=0.12, right=0.88, bottom=0.22, top=0.90)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    plt.close(figure)
    return np.asarray(Image.fromarray(image).resize((width, height)))


def trace_image(rewards, barriers, index, width=640, height=220):
    """
    Draw reward and h from the first step through index.

    Arguments:
        rewards: reward at each step.
        barriers: clearance of the cart and pole at each step.
        index: last step to draw. Later steps stay empty.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        RGB array, shape (height, width, 3).
    """
    reward = np.asarray(rewards, dtype=float).reshape(-1)
    barrier = np.asarray(barriers, dtype=float).reshape(-1)
    last = int(index)
    steps = np.arange(last + 1)
    reward_limits, barrier_limits, step_limits = axis_limits(reward, barrier)
    figure, reward_axis = _draw(width, height)
    barrier_axis = reward_axis.twinx()
    barrier_axis.axhspan(0.0, barrier_limits[1], color="#1b9e4b", alpha=0.08, zorder=0)
    barrier_axis.axhline(0.0, color="#8f1d1d", linewidth=1.15, linestyle=(0, (5, 3)), zorder=2)
    reward_axis.plot(steps, reward[: last + 1], color="#1b9e4b", linewidth=1.6, zorder=3)
    barrier_axis.plot(steps, barrier[: last + 1], color="#d64545", linewidth=1.6, zorder=3)
    reward_axis.set_xlim(*step_limits)
    reward_axis.set_ylim(*reward_limits)
    barrier_axis.set_ylim(*barrier_limits)
    reward_axis.set_xlabel("step")
    reward_axis.set_ylabel("reward", color="#1b9e4b")
    barrier_axis.set_ylabel("h", color="#d64545")
    reward_axis.tick_params(axis="y", colors="#1b9e4b")
    barrier_axis.tick_params(axis="y", colors="#d64545")
    return _rgb(figure, width, height)


def psi_image(psi1, psi2, index, width=640, height=220):
    """
    Draw psi1 and psi2 from the first step through index.

    Arguments:
        psi1: speed constraint at each step.
        psi2: force constraint at each step.
        index: last step to draw. Later steps stay empty.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        RGB array, shape (height, width, 3).
    """
    first = np.asarray(psi1, dtype=float).reshape(-1)
    second = np.asarray(psi2, dtype=float).reshape(-1)
    last = int(index)
    steps = np.arange(last + 1)
    left_limits, right_limits = psi_limits(first, second)
    figure, left = _draw(width, height)
    right = left.twinx()
    left.axhline(0.0, color="#2c6bed", linewidth=1.15, linestyle=(0, (5, 3)), zorder=2)
    right.axhline(0.0, color="#e07a14", linewidth=1.15, linestyle=(0, (1, 2)), zorder=2)
    left.plot(steps, first[: last + 1], color="#2c6bed", linewidth=1.6, zorder=3, label=r"$\psi_1$")
    right.plot(steps, second[: last + 1], color="#e07a14", linewidth=1.6, zorder=3, label=r"$\psi_2$")
    left.set_xlim(*STEP_LIMITS)
    left.set_ylim(*left_limits)
    right.set_ylim(*right_limits)
    left.set_xlabel("step")
    left.set_ylabel(r"$\psi_1$", color="#2c6bed")
    right.set_ylabel(r"$\psi_2$", color="#e07a14")
    left.tick_params(axis="y", colors="#2c6bed")
    right.tick_params(axis="y", colors="#e07a14")
    left.legend(loc="upper left", frameon=False, fontsize=8)
    right.legend(loc="upper right", frameon=False, fontsize=8)
    return _rgb(figure, width, height)


def frames_with_trace(frames, rewards, barriers, plot_height=220, psi1=None, psi2=None):
    """
    Place the growing traces under each cart-pole frame.

    Arguments:
        frames: rendered frames, one per step.
        rewards: reward at each step.
        barriers: real safety margin h at each step.
        plot_height: height of each trace strip.
        psi1: speed constraint at each step. Drawn only together with psi2.
        psi2: force constraint at each step.

    Returns:
        List of RGB frames, each taller than the cart-pole frame.
    """
    if (psi1 is None) != (psi2 is None):
        raise ValueError("psi1 and psi2 are drawn together.")
    width = int(np.asarray(frames[0]).shape[1])
    stacked = []
    for index, frame in enumerate(frames):
        parts = [np.asarray(frame), trace_image(rewards, barriers, index, width=width, height=plot_height)]
        if psi1 is not None:
            parts.append(psi_image(psi1, psi2, index, width=width, height=plot_height))
        stacked.append(np.concatenate(parts, axis=0))
    return stacked
