#!/usr/bin/env bash
# Add ParaStell (and pymoab) to the stellcoilbench_vmec conda environment.
#
# ParaStell is required for FEM structural analysis. Run with stellcoilbench_vmec active.
#
# Usage:
#   conda activate stellcoilbench_vmec
#   bash tools/install_parastell_in_vmec.sh

set -e

if [ -z "${CONDA_PREFIX}" ]; then
    echo "ERROR: No conda environment active. Run: conda activate stellcoilbench_vmec"
    exit 1
fi

if python -c "from parastell.magnet_coils import MagnetSetFromFilaments" 2>/dev/null; then
    echo "ParaStell already installed. Skipping."
    exit 0
fi

echo "Installing ParaStell into ${CONDA_PREFIX}..."

echo "[1/4] conda: moab, cadquery, gmsh, cad_to_dagmc"
conda install -y -c conda-forge moab cadquery gmsh "cad_to_dagmc>=0.9.1"

echo "[2/4] pip: Cython 0.29.x"
pip install "cython>=0.29,<3"

echo "[3/4] pip: pymoab (patched for MOAB 5.6 + pydagmc compatibility)"
WORK_DIR=$(mktemp -d)
trap "rm -rf ${WORK_DIR}" EXIT

git clone --depth 1 https://github.com/scopatz/pymoab.git "${WORK_DIR}/pymoab"
git clone --depth 1 --branch 5.5.1 https://bitbucket.org/fathomteam/moab.git "${WORK_DIR}/moab"

mkdir -p "${WORK_DIR}/moab_compat/include/moab"
cp "${WORK_DIR}/moab/src/TagInfo.hpp" "${WORK_DIR}/moab_compat/include/moab/"

# Remove include_dirs from cythonize() - Cython 3 rejects it
python3 -c "
import re
p = \"${WORK_DIR}/pymoab/setup.py\"
s = open(p).read()
s = re.sub(r',\s*include_dirs=include_path\s*', ' ', s)
s = re.sub(r',\s*\)', ')', s)
open(p, 'w').write(s)
"

# pydagmc expects 'rng' but pymoab has 'range'
cat > "${WORK_DIR}/pymoab/pymoab/__init__.py" << 'INIT'
from . import core, types, tag
from . import range as rng
INIT

CFLAGS="-I${WORK_DIR}/moab_compat/include" \
MOAB_PATH="${CONDA_PREFIX}" \
pip install --no-build-isolation "${WORK_DIR}/pymoab"

echo "[4/4] pip: pydagmc, parastell"
pip install pydagmc
pip install --no-deps "parastell @ git+https://github.com/svalinn/parastell.git"

echo ""
if python -c "from parastell.magnet_coils import MagnetSetFromFilaments" 2>/dev/null; then
    echo "SUCCESS: ParaStell installed. Structural analysis should work."
else
    echo "WARNING: ParaStell import check failed."
    exit 1
fi
