"""
Load the released swing-up network, dynamics, reward, and real cart-pole.
"""
import json
import os
import pickle
from pathlib import Path


def load_swingup(trial_dir, check_idx=1150):
    """
    Load checkpoint files and restore the network.

    Arguments:
        trial_dir: directory that contains params.json and checkpoint_XXXXXX.
        check_idx: checkpoint index. 1150 is the main swing-up network.

    Returns:
        Dict with env, real_env, policy, dyn_model, rew_model, and checkpoint_dir.

    Raises:
        FileNotFoundError: a required path is missing. The message includes that path.
    """
    trial_dir = Path(trial_dir)
    if not trial_dir.is_dir():
        raise FileNotFoundError(trial_dir)
    checkpoint_dir = trial_dir / f"checkpoint_{check_idx:06d}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    params_path = trial_dir / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(params_path)
    for name in ("dyn_model.pkl", "rew_model.pkl"):
        model_path = checkpoint_dir / name
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    from safesindy.preload import install_hydrogym_stub

    install_hydrogym_stub()

    import ray
    from ray.rllib.algorithms import Algorithm

    from sindy_rl.env import BaseEnsembleSurrogateEnv
    from sindy_rl.policy import RLlibPolicyWrapper

    _import_saved_classes()

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                }
            },
        )

    with params_path.open() as handle:
        config = json.load(handle)
    env = _env_from_config(config, BaseEnsembleSurrogateEnv)
    with (checkpoint_dir / "dyn_model.pkl").open("rb") as handle:
        dyn_model = pickle.load(handle)
    with (checkpoint_dir / "rew_model.pkl").open("rb") as handle:
        rew_raw = pickle.load(handle)
    rew_model = _restore_reward(rew_raw, config)
    if not hasattr(dyn_model, "predict"):
        raise TypeError("dynamics model has no predict")
    if not hasattr(rew_model, "predict"):
        raise TypeError("reward model has no predict")
    _check_reward(rew_model)

    algorithm = _algorithm_on_driver(checkpoint_dir, Algorithm)
    policy = RLlibPolicyWrapper(algorithm)
    real_env = env.real_env
    if not real_env:
        real_env = env.init_real_env()
    return {
        "env": env,
        "real_env": real_env,
        "policy": policy,
        "dyn_model": dyn_model,
        "rew_model": rew_model,
        "checkpoint_dir": checkpoint_dir,
        "config": config,
    }


def _import_saved_classes():
    """
    Import the modules named inside the dynamics and reward pickles.
    """
    import sindy_rl.dynamics
    import sindy_rl.reward

    return sindy_rl.dynamics, sindy_rl.reward


def _algorithm_on_driver(checkpoint_dir, algorithm_cls):
    """
    Restore the network without remote rollout workers.

    The released config asks for two workers. Each one builds a cart-pole,
    and those worker processes exit on this machine before they report spaces.
    The episode only needs the policy on the driver.
    """
    from ray.rllib.utils.checkpoints import get_checkpoint_info

    checkpoint_info = get_checkpoint_info(str(checkpoint_dir))
    state = algorithm_cls._checkpoint_info_to_algorithm_state(checkpoint_info)
    config = state["config"].copy(copy_frozen=False)
    config.num_rollout_workers = 0
    state["config"] = config
    # The connector blob is a Python 3.9 cloudpickle. The weights are plain arrays.
    # A fresh connector is enough for this feedforward policy.
    for policy_state in state["worker"]["policy_states"].values():
        policy_state.pop("connector_configs", None)
    return algorithm_cls.from_state(state)


def _restore_reward(raw, config):
    """
    Rebuild the reward model when the archive stored only its config dict.
    """
    if hasattr(raw, "predict"):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"reward checkpoint has type {type(raw).__name__}")
    from sindy_rl import reward as reward_module

    class_name = config["rew_model"]["class"]
    rew_class = getattr(reward_module, class_name)
    rew_config = raw if "name" in raw else config["rew_model"]["config"]
    return rew_class(rew_config)


def _check_reward(rew_model):
    import numpy as np

    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    action = np.array([0.0])
    value = rew_model.predict(state, action)
    if not np.isscalar(value):
        value = np.asarray(value).reshape(-1)[0]
    if not np.isfinite(value):
        raise ValueError("reward model returned a non-finite value")


def _env_from_config(config, env_class):
    env_config = config["drl"]["config"]["environment"]["env_config"]
    env_config["dynamics_model_config"] = config["dynamics_model"]
    env_config["rew_model_config"] = config["rew_model"]
    return env_class(env_config)
