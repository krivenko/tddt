#!/usr/bin/env bash
# D (flat-bath half-bandwidth) sweep for the Mickey-Mouse 1 model.
# Reads base parameters from param.in, overrides --D and --output_name for each run.
# Usage (from repo root):
#   bash programs/run_D_sweep.sh
#   MPI_NP=8 bash programs/run_D_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PARAM="$SCRIPT_DIR/param.in"
MPI_NP="${MPI_NP:-1}"

D_VALUES=(0.5 1.0 1.5 2.0 2.5 3.0)

for D in "${D_VALUES[@]}"; do
    LABEL="D${D}"
    TMP_PARAM="${TMPDIR:-/tmp}/param_mickey_$$_${D}.in"

    # Copy base params, stripping --D and --output_name lines, then append overrides.
    grep -v -E '^\s*--D\b|^\s*--output_name\b' "$BASE_PARAM" > "$TMP_PARAM"
    echo "--D           $D"            >> "$TMP_PARAM"
    echo "--output_name dtrimp_${LABEL}" >> "$TMP_PARAM"

    echo "=== Running D=$D  (out: dtrimp_${LABEL}.h5) ==="
    if [ "$MPI_NP" -gt 1 ]; then
        mpirun -n "$MPI_NP" python "$SCRIPT_DIR/dtrimp_run.py" "$TMP_PARAM"
    else
        python "$SCRIPT_DIR/dtrimp_run.py" "$TMP_PARAM"
    fi

    rm -f "$TMP_PARAM"
done

echo "=== D sweep complete ==="
