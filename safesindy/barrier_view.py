"""
Front view of the cart-pole with the forbidden region(s) shaded.
"""
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pathlib import Path

import numpy as np
from dm_control import mjcf
import dm_control.suite

from safesindy.barrier import right_barrier

FRAME_HEIGHT = 480
FRAME_WIDTH = 640
_XML = Path(dm_control.suite.__file__).resolve().parent / "cartpole.xml"


def render_barrier_frame(obs, physics=None, right_limit=None, left_limit=None):
    """
    Render one front-view frame.

    Arguments:
        obs: observation, cart position first, then cos, sin, and the two rates.
        physics: optional compiled model. A new one is built when omitted.
        right_limit, left_limit: forwarded to barrier_physics when physics is
            omitted; ignored otherwise (physics already has its slabs baked in).

    Returns:
        RGB array, shape (480, 640, 3), dtype uint8.
    """
    if physics is None:
        if right_limit is None and left_limit is None:
            right_limit = right_barrier()
        physics = barrier_physics(right_limit=right_limit, left_limit=left_limit)
    _set_state(physics, obs)
    return np.asarray(physics.render(height=FRAME_HEIGHT, width=FRAME_WIDTH, camera_id="fixed"))


def render_barrier_frames(observations, right_limit=None, left_limit=None):
    """
    Render one front-view frame per observation.

    Arguments:
        observations: sequence of observations.
        right_limit: world x where the forbidden right-hand region starts.
            Defaults to right_barrier() if left_limit is also omitted
            (matches the original single-barrier behavior); otherwise must
            be given explicitly if a right slab is wanted.
        left_limit: world x where the forbidden left-hand region starts
            (a cart or pole point further left than this is forbidden).
            Omit for no left slab.

    Returns:
        List of RGB frames.
    """
    if right_limit is None and left_limit is None:
        right_limit = right_barrier()
    physics = barrier_physics(right_limit=right_limit, left_limit=left_limit)
    return [render_barrier_frame(obs, physics) for obs in observations]


def barrier_physics(right_limit=None, left_limit=None):
    """
    Cart-pole model with a translucent red slab over each forbidden region.

    Each slab starts at its barrier position and continues past the
    corresponding edge of the front camera's view, covering its full height.
    """
    far = 4.0
    model = mjcf.from_path(str(_XML))
    if right_limit is not None:
        model.worldbody.add(
            "geom",
            name="barrier_right",
            type="box",
            pos=[0.5 * (right_limit + far), 0.0, 1.5],
            size=[0.5 * (far - right_limit), 0.02, 3.0],
            rgba=[1.0, 0.15, 0.15, 0.28],
            contype=0,
            conaffinity=0,
        )
    if left_limit is not None:
        model.worldbody.add(
            "geom",
            name="barrier_left",
            type="box",
            pos=[0.5 * (left_limit - far), 0.0, 1.5],
            size=[0.5 * (left_limit - (-far)), 0.02, 3.0],
            rgba=[1.0, 0.15, 0.15, 0.28],
            contype=0,
            conaffinity=0,
        )
    return mjcf.Physics.from_mjcf_model(model)


def _set_state(physics, obs):
    flat = np.asarray(obs, dtype=float).reshape(-1)
    with physics.reset_context():
        physics.named.data.qpos["slider"] = flat[0]
        physics.named.data.qpos["hinge_1"] = np.arctan2(flat[2], flat[1])
        physics.named.data.qvel["slider"] = flat[3]
        physics.named.data.qvel["hinge_1"] = flat[4]
