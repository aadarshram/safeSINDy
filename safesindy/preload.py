"""
Let sindy_rl import on a machine that does not have Hydrogym.
"""
import os
import sys
from pathlib import Path

STUB_ROOT = Path(__file__).resolve().parents[1] / "stubs"


def install_hydrogym_stub():
    """
    Put the Hydrogym stand-in on the import path for this process and Ray workers.

    sindy_rl.registry subclasses FlowEnv at import time. The cart-pole path
    never constructs that class. Ray workers start a fresh interpreter, so the
    stand-in has to be a real module, not only an object inserted in this process.
    """
    stub_root = str(STUB_ROOT)
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if stub_root not in entries and STUB_ROOT.is_dir():
        os.environ["PYTHONPATH"] = stub_root if not current else stub_root + os.pathsep + current
    if stub_root not in sys.path and STUB_ROOT.is_dir():
        sys.path.insert(0, stub_root)
    import hydrogym  # noqa: F401
