"""
Fit the swing-up sparse policy the way sindy-rl/sindy_rl/sparse_policy/cart.py does.

That script keeps the fit under `if __name__`, so the numbers are repeated here.
The regression itself uses the same optimizer, library, and ensemble settings.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from pysindy import PolynomialLibrary
from tqdm import tqdm

from safesindy.preload import install_hydrogym_stub

install_hydrogym_stub()

from sindy_rl.env import rollout_env
from sindy_rl.policy import RLlibPolicyWrapper, SparseEnsemblePolicy
from sindy_rl.sindy_utils import build_optimizer
from safesindy.load import load_swingup
from safesindy.sparse_policy import format_expression, save_sparse_policy

N_INITIAL = 400
INITIAL_POSITION_SCALE = 0.25
INITIAL_ANGLE_LOC = np.pi
INITIAL_ANGLE_SCALE = 0.1
N_DYN_ROLLOUT = 5000
STEPS_BETWEEN_RESETS = 500
N_RESAMPLES = 2
NOISE_SCALE = 1e-1
TEACHER_CLIP = 5.0
POLICY_CLIP = 1.0
POLY_DEGREE = 3
N_MODELS = 20
THRESHOLDS = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
INPUT_FEATURES = ["x", "cos_theta", "sin_theta", "dx", "dtheta"]
CLIP_BOUNDS = np.array([[-POLICY_CLIP], [POLICY_CLIP]])


def fit_and_save(trial_dir, out_dir, check_idx=1150):
    """
    Load the checkpoint, fit the cubic policy, and write npz and json.

    Arguments:
        trial_dir: released swing-up trial directory.
        out_dir: directory for cart_swingup_<index>.json and the npz.
        check_idx: checkpoint index.

    Returns:
        Path to the json file.
    """
    loaded = load_swingup(trial_dir, check_idx=check_idx)
    dyn_model = loaded["dyn_model"]
    rew_model = loaded["rew_model"]
    print(f"dynamics model: {type(dyn_model).__name__}")
    print(f"reward model: {type(rew_model).__name__}")
    if not hasattr(rew_model, "predict") or not hasattr(dyn_model, "predict"):
        raise TypeError("dynamics and reward models must provide predict")

    # Collect noisy states and the network's actions on those states.
    unbounded = RLlibPolicyWrapper(loaded["policy"].algo.get_policy(), mode="policy")
    arrays = teacher_arrays(
        loaded["env"],
        dyn_model,
        loaded["policy"],
        unbounded,
    )

    # Select the threshold pair by mean-coefficient error, then keep the mean.
    policy = select_mean_policy(*arrays)
    coef = _coef_row(policy)
    feature_names = list(policy.feature_library.get_feature_names(input_features=INPUT_FEATURES))
    expression = format_expression(coef, feature_names)
    print(expression)

    json_path = Path(out_dir) / f"cart_swingup_{check_idx}.json"
    save_sparse_policy(
        json_path,
        coef,
        {
            "checkpoint": check_idx,
            "poly_degree": POLY_DEGREE,
            "include_bias": False,
            "include_interaction": True,
            "input_features": INPUT_FEATURES,
            "feature_names": feature_names,
            "min_bounds": [-POLICY_CLIP],
            "max_bounds": [POLICY_CLIP],
            "expression": expression,
        },
    )
    return json_path


def teacher_arrays(env, dyn_model, policy, unbounded):
    """
    Build the regression arrays used by the cart-pole script.

    Arguments:
        env: surrogate environment.
        dyn_model: unpickled ensemble dynamics.
        policy: clipped network wrapper used to roll the surrogate.
        unbounded: unclipped network, queried for the regression labels.

    Returns:
        X_data, U_data, X_eval, U_eval.
    """
    # Draw initial states near the hanging equilibrium.
    np.random.seed(0)
    angle = np.random.normal(loc=INITIAL_ANGLE_LOC, scale=INITIAL_ANGLE_SCALE, size=N_INITIAL)
    initial = np.random.normal(loc=0, scale=INITIAL_POSITION_SCALE, size=(N_INITIAL, 5))
    initial[:, 1] = np.cos(angle)
    initial[:, 2] = np.sin(angle)

    # Roll the median dynamics model from those states.
    env.dynamics_model = deepcopy(dyn_model)
    env.dynamics_model.set_median_coef_()
    env.use_real_env = False
    env.reset_from_buffer = True
    env.buffer_dict = {"x": [initial]}
    env.reset()
    trajectories, _actions, _rewards = rollout_env(
        env,
        policy,
        n_steps=N_DYN_ROLLOUT,
        n_steps_reset=STEPS_BETWEEN_RESETS,
        verbose=True,
    )

    # Repeat the states, add noise, and query the unclipped network.
    np.random.seed(1)
    clean = np.concatenate(trajectories)
    repeated = np.concatenate([clean for _ in range(N_RESAMPLES)])
    noise = np.random.normal(loc=0, scale=NOISE_SCALE, size=repeated.shape)
    x_data = repeated + noise
    u_data = _clipped_actions(unbounded, x_data)
    x_eval = clean
    u_eval = _clipped_actions(unbounded, x_eval)
    return x_data, u_data, x_eval, u_eval


def select_mean_policy(x_data, u_data, x_eval, u_eval):
    """
    Sweep the threshold grid and return the policy at its ensemble mean.

    Model choice uses the mean coefficients. The returned policy is also set
    to that mean, which is the controller the reference script rolls out.
    """
    mesh = np.meshgrid(THRESHOLDS, THRESHOLDS)
    hyper_mesh = np.array([mesh[0].flatten(), mesh[1].flatten()]).T
    policies = []
    for alpha, threshold in tqdm(hyper_mesh):
        policies.append(
            _fit_one(
                x_data,
                u_data,
                alpha=alpha,
                threshold=threshold,
            )
        )
    scores = np.array(
        [
            _score(policy, x_eval, u_eval, use_median=False)
            for policy in policies
        ]
    )
    best = policies[int(np.argmin(scores[:, 0]))]
    best.set_mean_coef_()
    return best


def _fit_one(x_data, u_data, alpha, threshold):
    optimizer = build_optimizer(
        {
            "base_optimizer": {
                "name": "STLSQ",
                "kwargs": {"alpha": float(alpha), "threshold": float(threshold)},
            },
            "ensemble": {
                "bagging": True,
                "library_ensemble": True,
                "n_models": N_MODELS,
            },
        }
    )
    library = PolynomialLibrary(
        degree=POLY_DEGREE,
        include_bias=False,
        include_interaction=True,
    )
    policy = SparseEnsemblePolicy(
        optimizer=optimizer,
        feature_library=library,
        min_bounds=CLIP_BOUNDS[0],
        max_bounds=CLIP_BOUNDS[1],
    )
    policy.fit([x_data], [u_data])
    return policy


def _score(policy, x_eval, u_eval, use_median):
    if use_median:
        policy.set_median_coef_()
    else:
        policy.set_mean_coef_()
    theta = policy.feature_library.transform(x_eval)
    pred = (policy.optimizer.coef_ @ theta.T).T
    pred = np.clip(pred, *CLIP_BOUNDS)
    target = np.clip(u_eval, *CLIP_BOUNDS)
    mse = np.mean((pred - target) ** 2)
    penalty = np.abs(policy.optimizer.coef_).sum()
    return mse, penalty


def _clipped_actions(unbounded, states):
    actions = unbounded.algo.compute_actions(states, explore=False)[0]
    return np.clip(actions, -TEACHER_CLIP, TEACHER_CLIP)


def _coef_row(policy):
    coef = np.atleast_2d(np.asarray(policy.optimizer.coef_, dtype=float))
    n_terms = policy.feature_library.transform(np.zeros((1, len(INPUT_FEATURES)))).shape[1]
    if coef.shape[-1] != n_terms and coef.shape[0] == n_terms:
        coef = coef.T
    if coef.shape[-1] != n_terms:
        raise ValueError(f"coefficient shape {coef.shape} does not match {n_terms} terms")
    return coef
