#!/bin/bash
# Source this before building or evaluating Brevis on the Nibi cluster.
#   source setup_env.sh
#
# Zig is not packaged as an Lmod module on Alliance clusters, so it is a local
# toolchain install. Override ZIG_ROOT if you keep it elsewhere.
#
# Do NOT export OMP_NUM_THREADS (or GZIP/XZ_OPT/ZSTD_NBTHREADS/...) here. They
# are in eval/benchmarking.py's CODEC_ENVIRONMENT_VARIABLES, and the formal
# campaign gate requires them to be truly unset. Pass thread counts with
# --jobs instead.

module purge
module load StdEnv/2023 python/3.12

BREVIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Zig lookup order: explicit ZIG_ROOT, then a project-local toolchain, then the
# per-user install. .toolchain/ is Git-ignored.
if [ -z "$ZIG_ROOT" ]; then
    for candidate in "$BREVIS_ROOT/.toolchain/zig-0.16.0" "$HOME/software/zig-0.16.0"; do
        [ -x "$candidate/zig" ] && ZIG_ROOT="$candidate" && break
    done
fi
if [ -x "$ZIG_ROOT/zig" ]; then
    export PATH="$ZIG_ROOT:$PATH"
else
    echo "setup_env.sh: no zig found — see README (Cluster Setup)" >&2
fi

# Python needs only numpy, and only for eval/tensor_stats.py. Keep the venv on
# the node-local disk; the parallel filesystem is slow on many small files.
#
# Some Alliance sessions (JupyterHub in particular) export PIP_PREFIX and
# PYTHONPATH. PIP_PREFIX silently redirects installs out of the active venv --
# pip still reports success -- and PYTHONPATH leaks cvmfs site-packages into
# it. Clear both so the venv is the only source of packages.
unset PIP_PREFIX PYTHONPATH

BREVIS_VENV="${BREVIS_VENV:-${SLURM_TMPDIR:-/tmp}/brevis-venv}"
if [ ! -x "$BREVIS_VENV/bin/python" ]; then
    virtualenv --no-download "$BREVIS_VENV" >/dev/null &&
        "$BREVIS_VENV/bin/pip" install --no-index --quiet \
            -r "$BREVIS_ROOT/requirements.txt"
fi
# shellcheck disable=SC1091
[ -f "$BREVIS_VENV/bin/activate" ] && source "$BREVIS_VENV/bin/activate"

# Keep the Zig build cache off the parallel filesystem's small-file path when a
# job-local disk is available.
if [ -n "$SLURM_TMPDIR" ]; then
    export ZIG_GLOBAL_CACHE_DIR="$SLURM_TMPDIR/zig-cache"
fi
