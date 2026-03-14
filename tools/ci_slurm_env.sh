#!/usr/bin/env bash
# Shared environment for SLURM job scripts on Viper compute nodes.
# Source from generated batch scripts: source __WORKSPACE__/tools/ci_slurm_env.sh
#
# Sets up: modules (intel, impi, python-waterboa), venv, MPLBACKEND, PYTHONUNBUFFERED.
# Optionally exports STELLCOILBENCH_CI_VERBOSE_STDOUT=1 when STELLCOILBENCH_CI_VERBOSE is set.
set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load intel/2024.0 impi/2021.11
export LD_LIBRARY_PATH="${I_MPI_ROOT:-}/lib/release:${LD_LIBRARY_PATH:-}"
module load python-waterboa/2025.06
source "${HOME:-}/stellcoilbench-env/bin/activate"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
[ -n "${STELLCOILBENCH_CI_VERBOSE:-}" ] && export STELLCOILBENCH_CI_VERBOSE_STDOUT=1
