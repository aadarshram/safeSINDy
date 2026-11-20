"""
State-dependent safety margin from the SINDy ensemble's own fitted
uncertainty, following SINDy-RL's own pointwise-variance formula
(Zolman et al., Eq. 6-7): for a dictionary model y(x) = Theta(x) xi, the
predictive variance at a query point is Theta(x) Cov(xi) Theta(x)^T.

Ensemble members are trimmed first via a robust outlier rule (deviation
from the ensemble median beyond median + k*MAD), since the raw ensemble
is dominated by one badly divergent, never-validated member.
"""
import numpy as np

POLE_LENGTH = 1.0


def trim_ensemble(coef_list, library, feature_states, feature_actions, mad_k=5.0):
    """
    Indices of ensemble members to KEEP, by robust outlier rejection on
    one-step prediction deviation from the ensemble median.

    Arguments:
        coef_list: ensemble coefficients, shape (n_models, n_state, n_features).
        library: fitted feature library (same one used to train coef_list).
        feature_states, feature_actions: sample states/actions (e.g. a real
            rollout) used to measure each member's deviation from the median.
        mad_k: outlier threshold in units of median absolute deviation.

    Returns:
        List of member indices to keep.
    """
    coef_list = np.asarray(coef_list, dtype=float)
    median_coef = np.median(coef_list, axis=0)
    n_models = coef_list.shape[0]
    deviation = np.zeros(n_models)
    for state, action in zip(feature_states, feature_actions):
        packed = np.concatenate([state, action]).reshape(1, -1)
        theta = library.transform(packed).reshape(-1)
        preds = coef_list @ theta
        median_pred = median_coef @ theta
        deviation += np.sum(np.abs(preds - median_pred), axis=1)
    deviation /= max(len(feature_states), 1)

    median_dev = np.median(deviation)
    mad = np.median(np.abs(deviation - median_dev))
    threshold = median_dev + mad_k * mad
    return [i for i in range(n_models) if deviation[i] <= threshold]


class VarianceInflation:
    """
    Adaptive inflation: max(floor, k * predicted std of body extent),
    using the trimmed ensemble's per-output coefficient covariance.
    """

    def __init__(self, coef_list, library, keep_idx, floor=0.05, k=3.0):
        coef_list = np.asarray(coef_list, dtype=float)[keep_idx]
        # Cov(xi_i) per output row i, shape (n_state, n_features, n_features)
        self.cov_rows = np.array(
            [np.cov(coef_list[:, i, :], rowvar=False) for i in range(coef_list.shape[1])]
        )
        self.library = library
        self.floor = float(floor)
        self.k = float(k)

    def __call__(self, state, force=0.0):
        """
        Arguments:
            state: current observation, shape (5,): [x, cos, sin, v, rate].
            force: candidate force (the variance estimate is mild in its
                dependence on force here; the current state dominates it).

        Returns:
            Inflation value, in meters.
        """
        packed = np.concatenate([np.asarray(state, dtype=float).reshape(-1), [float(force)]]).reshape(1, -1)
        theta = self.library.transform(packed).reshape(-1)
        var_pos = theta @ self.cov_rows[0] @ theta
        var_sin = theta @ self.cov_rows[2] @ theta
        std_extent = float(np.sqrt(max(var_pos, 0.0) + (POLE_LENGTH ** 2) * max(var_sin, 0.0)))
        return max(self.floor, self.k * std_extent)
