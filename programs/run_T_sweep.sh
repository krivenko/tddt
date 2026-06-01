#!/usr/bin/env bash
# Temperature sweep for the U-bath-U dimer model.
# Reads base parameters from param_dimer.in, overrides --T and --out_name for each run.
# Usage (from repo root):
#   bash programs/run_T_sweep.sh
#   MPI_NP=8 bash programs/run_T_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PARAM="$SCRIPT_DIR/param_dimer.in"
MPI_NP="${MPI_NP:-1}"

TEMPERATURES=(0 3 10)

for T in "${TEMPERATURES[@]}"; do
    LABEL="T${T}"
    TMP_PARAM="$(mktemp /tmp/param_dimer_XXXXXX.in)"

    # Copy base params, stripping --T and --out_name lines, then append overrides.
    grep -v -E '^\s*--T\b|^\s*--out_name\b' "$BASE_PARAM" > "$TMP_PARAM"
    echo "--T        $T"                          >> "$TMP_PARAM"
    echo "--out_name hubbard_bath_dimer_${LABEL}" >> "$TMP_PARAM"

    echo "=== Running T=$T  (out: hubbard_bath_dimer_${LABEL}) ==="
    if [ "$MPI_NP" -gt 1 ]; then
        mpirun -n "$MPI_NP" python programs/hubbard_dimer_chi.py "$TMP_PARAM"
    else
        python programs/hubbard_dimer_chi.py "$TMP_PARAM"
    fi

    rm -f "$TMP_PARAM"
done

echo "=== T sweep complete ==="
