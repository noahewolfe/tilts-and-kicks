#!/usr/bin/env bash
# =============================================================================
# setup-env-260228.sh
# Reproducible setup for the just-for-kicks-260228 mamba environment.
#
# Prerequisites:
#   - mamba is available on PATH
#   - jax 0.9.0 is already installed in the environment
#   - The local software repos exist under:
#       <project_root>/software/gwpopulation
#       <project_root>/software/pixelpop
#
# Usage:
#   cd /home/submit/newolfe/projects/tilts-and-kicks
#   bash setup-env-260228.sh
# =============================================================================

set -euo pipefail

ENV_NAME="just-for-kicks-260228"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PYTHON="/work/submit/newolfe/miniforge3/envs/${ENV_NAME}/bin/python"
ENV_PIP="${ENV_PYTHON} -m pip"
ENV_SITE_PACKAGES="$( mamba run -n "${ENV_NAME}" python -c \
    "import site; print(site.getsitepackages()[0])" )"

echo "=== Setting up ${ENV_NAME} ==="
echo "Project root : ${PROJECT_ROOT}"
echo "Site-packages: ${ENV_SITE_PACKAGES}"

# ---------------------------------------------------------------------------
# Step 1: Install gwpopulation from the local editable repo
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/5] Installing gwpopulation (editable) ---"
mamba run -n "${ENV_NAME}" pip install -e "${PROJECT_ROOT}/software/gwpopulation"

# ---------------------------------------------------------------------------
# Step 2: Install pixelpop from the local editable repo
#   (pulls in numpyro, gwpopulation_pipe, bilby_pipe, lalsuite, etc.)
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/5] Installing pixelpop (editable) ---"
mamba run -n "${ENV_NAME}" pip install -e "${PROJECT_ROOT}/software/pixelpop"

# ---------------------------------------------------------------------------
# Step 3: Install htcondor via conda-forge
#
#   The PyPI htcondor stub (installed as a side-effect of gwpopulation_pipe)
#   cannot be imported because it lacks the real system libraries.
#   The conda-forge package provides working binaries but exposes the Python
#   module under the NEW name `htcondor2` (changed in HTCondor >= 23.x).
#   gwpopulation_pipe still uses `import htcondor` (old name), so we:
#     a) Remove the broken pip stub.
#     b) Install the real conda-forge package.
#     c) Drop a thin compatibility shim that aliases htcondor -> htcondor2.
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/5] Fixing htcondor (conda-forge + compatibility shim) ---"

# 3a. Remove the non-functional pip stub if present
mamba run -n "${ENV_NAME}" pip uninstall -y htcondor 2>/dev/null || true

# 3b. Install from conda-forge (provides htcondor2 module + system libs)
mamba install -n "${ENV_NAME}" -c conda-forge htcondor -y

# 3c. Write the htcondor -> htcondor2 compatibility shim
SHIM_DIR="${ENV_SITE_PACKAGES}/htcondor"
mkdir -p "${SHIM_DIR}"
cat > "${SHIM_DIR}/__init__.py" << 'PYEOF'
# Compatibility shim: htcondor -> htcondor2
# The conda-forge htcondor package >=23.x renamed the Python module from
# 'htcondor' to 'htcondor2'. This shim re-exports everything from htcondor2
# so that code using the old name continues to work.
from htcondor2 import *  # noqa: F401, F403
from htcondor2 import dags  # noqa: F401
import htcondor2 as _htcondor2

# Re-export all public names explicitly
import sys as _sys
_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _htcondor2.__dict__.items() if not k.startswith('__')}
)
PYEOF

echo "htcondor shim written to ${SHIM_DIR}/__init__.py"

# Verify the shim works
mamba run -n "${ENV_NAME}" python -c \
    "import htcondor; from htcondor import dags; print('htcondor shim OK')"

# ---------------------------------------------------------------------------
# Step 4: Patch optype for Python 3.11 compatibility
#
#   optype 0.13.4 uses PEP 696 TypeVar defaults (e.g. `default=int`) which
#   are only honoured by the typing machinery in Python 3.13+.  Under
#   Python 3.11, _check_generic() enforces the exact parameter count, so
#   `CanSequence[CanIndex, _ValT]` (2 args) fails because CanSequence is
#   Protocol[_IndexT_contra, _V_co, _IntT_co] (3 params).
#   Fix: supply the missing `int` default explicitly in the one call site
#   inside optype._core._does.
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/5] Patching optype for Python 3.11 (PEP 696 workaround) ---"

OPTYPE_DOES="${ENV_SITE_PACKAGES}/optype/_core/_does.py"
sed -i \
    's/sequence: _c\.CanSequence\[_c\.CanIndex, _ValT\],/sequence: _c.CanSequence[_c.CanIndex, _ValT, int],  # int = default for _IntT_co (PEP 696 not available in py3.11)/' \
    "${OPTYPE_DOES}"

# Verify the fix
mamba run -n "${ENV_NAME}" python -c "import unxt; print('unxt (optype patch) OK')"

# ---------------------------------------------------------------------------
# Step 5: Update run-multimass-gwpop.sh to use the -260228 environment
#   and guard against PYTHONPATH contamination from the JupyterHub spawner.
# ---------------------------------------------------------------------------
echo ""
echo "--- [5/5] Patching run-multimass-gwpop.sh ---"
SCRIPT="${PROJECT_ROOT}/code/strong/run-multimass-gwpop.sh"
sed -i "s|miniforge3/envs/just-for-kicks-260227/bin/python|miniforge3/envs/${ENV_NAME}/bin/python|g" \
    "${SCRIPT}"
# Add PYTHONPATH / PYTHONNOUSERSITE guards after the JAX_ENABLE_X64 line
# (idempotent: only inserts if not already present)
if ! grep -q "unset PYTHONPATH" "${SCRIPT}"; then
    sed -i '/export JAX_ENABLE_X64/a \\n# Prevent the JupyterHub spawner env from contaminating sys.path\nunset PYTHONPATH' "${SCRIPT}"
fi
echo "Patched: ${SCRIPT}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the inference:"
echo "  cd ${PROJECT_ROOT}/code/strong && bash run-multimass-gwpop.sh"
