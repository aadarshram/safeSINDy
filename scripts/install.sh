#!/usr/bin/env bash
# Install the cart-pole stack into the active safesindy environment.
# pysindy is pinned to 1.7.5 because the released fit code uses the 1.x
# EnsembleOptimizer API. numpy, scipy, scikit-learn, pyarrow, pandas, and
# setuptools are capped so those pins still import under Ray 2.6.
# Do not install Mesa or set MUJOCO_GL=osmesa.
# Video rendering uses EGL (MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python -m pip install \
  "ray[rllib]==2.6.3" \
  "pyarrow>=6.0.1,<16" \
  "pandas>=1.3,<2.3" \
  torch \
  "dm-env==1.6" \
  "dm_control==1.0.14" \
  "mujoco==2.3.7" \
  "numpy>=1.23,<2" \
  "scipy>=1.9,<1.14" \
  "pysindy==1.7.5" \
  "scikit-learn==1.3.2" \
  tqdm \
  imageio \
  imageio-ffmpeg \
  "setuptools<81" \
  pytest

python -m pip install -e "$root/../sindy-rl"
python -m pip install -e "$root"

# Their setup.py filters packages to names starting with "src", so the editable
# install records the project but does not put sindy_rl on the import path.
site="$(python -c 'import site; print(site.getsitepackages()[0])')"
sibling="$(cd "$root/../sindy-rl" && pwd)"
printf '%s\n' "$sibling" > "$site/sindy_rl_sibling.pth"
