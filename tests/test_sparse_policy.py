import json

import numpy as np
import pytest
from safesindy.sparse_policy import load_sparse_policy
from pysindy import PolynomialLibrary


def _feature_names():
    """
    Feature names for a degree-1 library on one input.
    """
    library = PolynomialLibrary(degree=1, include_bias=False, include_interaction=True)
    library.fit(np.zeros((2, 1)))
    return list(library.get_feature_names(input_features=["x"]))


def _write_linear_policy(directory):
    """
    Write u = clip(0.5 x, -1, 1) in the saved-policy format.
    """
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "linear.json"
    np.savez(directory / "linear.npz", coef=np.array([[0.5]]))
    meta = {
        "checkpoint": 1150,
        "poly_degree": 1,
        "include_bias": False,
        "include_interaction": True,
        "input_features": ["x"],
        "feature_names": _feature_names(),
        "min_bounds": [-1.0],
        "max_bounds": [1.0],
        "expression": "u0 = 0.500 x",
    }
    json_path.write_text(json.dumps(meta))
    return json_path


def test_linear_law_matches_clip(tmp_path):
    """
    A saved degree-1 law evaluates to the clipped product.
    """
    policy = load_sparse_policy(_write_linear_policy(tmp_path))
    unit = np.asarray(policy.compute_action(np.array([1.0]))).reshape(-1)[0]
    high = np.asarray(policy.compute_action(np.array([3.0]))).reshape(-1)[0]
    low = np.asarray(policy.compute_action(np.array([-4.0]))).reshape(-1)[0]
    assert unit == pytest.approx(0.5)
    assert high == pytest.approx(1.0)
    assert low == pytest.approx(-1.0)


def test_missing_policy_file_raises(tmp_path):
    """
    A missing policy file raises FileNotFoundError with that path.
    """
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_sparse_policy(missing)
    assert str(missing) in str(exc_info.value)
