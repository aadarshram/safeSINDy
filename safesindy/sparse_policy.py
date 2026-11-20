"""
Saved sparse policy: coefficients, library specification, and evaluation.
"""
import json
from pathlib import Path

import numpy as np
from pysindy import PolynomialLibrary


def format_expression(coef, feature_names, precision=3):
    """
    Format one printed line per action component.

    Arguments:
        coef: array shaped (n_actions, n_terms).
        feature_names: library names in the same order as the columns of coef.
        precision: digits after the decimal point.

    Returns:
        Multiline expression. A component with no surviving terms is 0.
    """
    lines = []
    for index, row in enumerate(np.atleast_2d(coef)):
        terms = []
        for weight, name in zip(row, feature_names):
            rounded = np.round(weight, precision)
            if rounded != 0:
                terms.append(f"{rounded:.{precision}f} {name}")
        body = " + ".join(terms) if terms else "0"
        lines.append(f"u{index} = {body}")
    return "\n".join(lines)


def save_sparse_policy(json_path, coef, meta):
    """
    Write the coefficient array and the json sidecar.

    Arguments:
        json_path: destination for the json. The npz uses the same stem.
        coef: coefficient array.
        meta: json-serializable library specification, including feature names.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    coef = np.atleast_2d(np.asarray(coef, dtype=float))
    np.savez(json_path.with_suffix(".npz"), coef=coef)
    json_path.write_text(json.dumps(meta, indent=2) + "\n")


def load_sparse_policy(json_path):
    """
    Load a sparse policy from its json sidecar and npz coefficients.

    Arguments:
        json_path: path to the json file.

    Returns:
        SparsePolicy

    Raises:
        FileNotFoundError: the json or the npz is missing. The message includes the path.
        ValueError: the rebuilt library's feature names do not match the file.
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    meta = json.loads(json_path.read_text())
    coef = np.load(npz_path)["coef"]
    return SparsePolicy(coef, meta)


class SparsePolicy:
    """
    Clipped polynomial policy u = clip(Theta(x) xi, min, max).
    """

    def __init__(self, coef, meta):
        self.coef = np.atleast_2d(np.asarray(coef, dtype=float))
        self.meta = meta
        self.expression = meta["expression"]
        self.min_bounds = np.asarray(meta["min_bounds"], dtype=float)
        self.max_bounds = np.asarray(meta["max_bounds"], dtype=float)
        self.input_features = list(meta["input_features"])
        self.library = PolynomialLibrary(
            degree=int(meta["poly_degree"]),
            include_bias=bool(meta["include_bias"]),
            include_interaction=bool(meta["include_interaction"]),
        )
        self.library.fit(np.zeros((1, len(self.input_features))))
        names = list(self.library.get_feature_names(input_features=self.input_features))
        expected = list(meta["feature_names"])
        if names != expected:
            raise ValueError(f"feature name mismatch: library {names} != file {expected}")

    def compute_action(self, obs):
        """
        Evaluate the polynomial and clip the action.

        Arguments:
            obs: observation vector.

        Returns:
            Action as float32, shape (n_actions,).
        """
        theta = self.library.transform(np.atleast_2d(np.asarray(obs, dtype=float)))
        action = (self.coef @ np.asarray(theta).T).reshape(-1)
        action = np.clip(action, self.min_bounds, self.max_bounds)
        return np.asarray(action, dtype=np.float32)
