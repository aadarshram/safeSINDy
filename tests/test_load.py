from safesindy.load import load_swingup


def test_missing_checkpoint_directory_raises(tmp_path):
    """
    A missing trial directory raises FileNotFoundError with that path.
    """
    missing = tmp_path / "no_such_trial"
    try:
        load_swingup(missing, check_idx=1150)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("load_swingup did not raise FileNotFoundError")
