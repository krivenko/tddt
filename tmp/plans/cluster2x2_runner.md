# Plan: scripts/ runner + output + notebook for tddt_2

## Context

`tddt_2/scripts/cluster2x2.py` is a working D-TRILEX script for a 2x2 plaquette cluster, but all parameters are hardcoded. The user wants a proper parameterized workflow in `scripts/`, matching `tddt` v1's `programs/` pattern:

- `mpirun -n 8 python scripts/cluster2x2_run.py scripts/param.in`
- HDF5 (default) or text-file output
- A plotting notebook at `scripts/plot_cluster2x2.ipynb`

**Environment**: `realevol3` conda env (installed at `/opt/anaconda3/envs/realevol3`).
Run command: `conda run -n realevol3 mpirun -n 8 python scripts/cluster2x2_run.py scripts/param.in`

---

## Files to Create

All files go into `scripts/`.

### 1. `scripts/param.in`

All physics parameters from `cluster2x2.py` exposed as CLI flags:

```
--t_max 10.0
--n_t 51
--n_ti 5001         # fine interpolation mesh (for TInterp)
--n_k 2             # k-points per dimension
--t_nn 1.0          # nearest-neighbor hopping
--t_nnn 0.0         # next-nearest-neighbor hopping
--A 0.0             # vector potential amplitude
--Omega 4.0         # laser frequency
--U 6.0             # Hubbard U before quench
--U1 4.0            # Hubbard U after quench
--V 0.2             # non-local interaction
--T 0.01            # temperature
--ex 0.0            # bath site 3 energy (absolute value)
--exx 3.0           # bath sites 1,2 energy ±exx (absolute)
--tx 0.7            # hopping to site 3 (absolute, = 0.7*t_nn in script)
--txx 1.0           # hopping to sites 1,2 (absolute, = 1.0*t_nn)
--output_dir data/cluster2x2
--output_name cluster2x2
#--simple_output
```

### 2. `scripts/cluster2x2_run.py`

Port of `cluster2x2.py` with parameterized inputs. Structure follows `tddt/programs/dtrimp_run.py`.

**MPI setup** (copy pattern from `dtrimp_run.py:56-76`):
- Try-import mpi4py, set `_rank`, `_size`, `_comm`
- `get_time()` uses `MPI.Wtime()` if available, else `time.perf_counter()`

**Argument parsing** (rank-0-reads-and-broadcasts, `dtrimp_run.py:260-275`):
- Rank 0 reads `sys.argv[1]` as param file, strips `#` comments, `shlex.split`, then `argparse.parse_args`
- `_comm.bcast(_args, root=0)` broadcasts to all ranks

**argparse parameters**:
```
--t_max, --n_t, --n_ti
--n_k, --t_nn, --t_nnn
--A, --Omega
--U, --U1, --V
--T
--ex, --exx, --tx, --txx    (plaquette bath params, absolute values)
--output_dir (default: "data/cluster2x2")
--output_name (default: "cluster2x2")
--simple_output              (store_true: write text files instead of HDF5)
```

**Computation** (same logic as `cluster2x2.py`, using parsed params):
1. Build `t_mesh = MeshReTime(0, t_max, n_t)`, `ti_mesh = MeshReTime(0, t_max, n_ti)`
2. Build `eps_tk`, `U_tq`, `U_dc` using parsed lattice/interaction params
3. Build `FiniteCluster` for reference system with parsed `ex, exx, tx, txx, U, T`
4. `theory = DualTRILEX(...)`, `compute_ref_init_state(T)`, apply quench (`U1`, `mu1`)
5. `compute_ref_correlators(...)`, `Delta = hybridization(...)`, `compute_bare_lines_vertex(...)`, `compute_diagrams()`
6. `g_tk = compute_lattice_gf()`, `g_cpt_tk = compute_lattice_gf_cpt()`
7. Compute `gd0_full_tk` and its Fourier transform `gd0_full_tr` (as in script lines 172-175)

**Timing**: print elapsed time after each major step (same pattern as `dtrimp_run.py:388-498`)

**Output** (rank 0 only):
- `os.makedirs(output_dir, exist_ok=True)`
- If `--simple_output`: `_write_simple(...)` — uses `write_keldysh_gf_file`
- Else: `_write_h5(...)` — HDF5 via h5py

**`_write_simple`**: write the same text files as `cluster2x2.py` (same filenames, same `write_keldysh_gf_file` format). Extract `write_keldysh_gf_file` into `scripts/utilities.py` and import it.

**`_write_h5`**: write HDF5 to `output_dir/output_name.h5`:
```
t_mesh                         # np.linspace(0, t_max, n_t)
ref/
  g_ref/FF, FB, BF, BB        # theory.g_ref
  g_imp/FF, FB, BF, BB        # theory.g_imp
dual/
  Sigma_loc/FF, FB, BF, BB    # local_part(theory.Sigma_tk)
  Gd0_loc/FF, FB, BF, BB      # local_part(theory.Gd0_reg_tk)
lattice/
  g_loc/FF, ...               # local_part(g_tk)
  g_cpt_loc/FF, ...           # local_part(g_cpt_tk)
  g_k0/FF, ..., g_k{N}/FF, ... # k-resolved g_tk
  g_cpt_k0/FF, ..., g_cpt_k{N}/...
  gd0_r0/FF, ..., gd0_r1/...  # gd0_full_tr at first two r-points
```

### 3. `scripts/utilities.py`

New file with I/O helpers:

```python
def write_keldysh_gf_file(filename, g, spc_point=None, target_indices=()):
    """Identical copy of the function from cluster2x2.py (lines 183-199)."""

def save_keldysh_gf_2pt_h5(h5group, name, g, spc_point=None, target_indices=(0,0,0,0)):
    """
    Save all 4 Keldysh components (FF, FB, BF, BB) to h5group[name].
    Iterates over all Branch combinations, writes g[b1,b2][t0,t,...][*target_indices]
    as a (n_t,) complex128 array (G(t, t'=0) slice for compact storage).
    Or write full g[b1,b2].data if that attribute is accessible.
    """
```

**Note on HDF5 saving**: At implementation time, check whether `g[b1, b2]` (where `g` is a `KeldyshGF` from tddt_2) exposes a `.data` numpy array. If yes, save that directly. If not, build the array from per-point access in the loop. Check `tddt_2/tddt/keldysh.py`.

### 4. `scripts/plot_cluster2x2.ipynb`

Notebook structure:

1. **Config cell**: parse `param.in` (same shlex approach), set `output_dir`, `output_name`, `simple_output` flag
2. **Load cell**: load from HDF5 (`h5py`) or text files into numpy arrays
3. **Plot: Reference system** — `G_ref(t, t'=0)` and `G_imp(t, t'=0)`: Re/Im of FW-FW and FW-BW
4. **Plot: Dual self-energy** — local Sigma(t, t'=0)
5. **Plot: Lattice GF** — `G_loc` vs `G_cpt_loc` (Re/Im)
6. **Plot: k-resolved** — one subplot per k-point for `G_k(t, t'=0)`
7. **Plot: Gd0** — bare dual propagator at r0, r1
8. Save all plots as PNG to `output_dir/plots/`

---

## Verification

```bash
conda activate realevol3

# install tddt_2 in dev mode (if not already)
cd /Users/gusein.bedirkhanov/Documents/PhD/FULL_DTRILEX_ON_KELDYSH/tddt_2
pip install -e .

# serial test
python scripts/cluster2x2_run.py scripts/param.in

# MPI run
mpirun -n 8 python scripts/cluster2x2_run.py scripts/param.in

# check output
ls data/cluster2x2/
# cluster2x2.h5 (or .txt files if --simple_output)

# open notebook
jupyter notebook scripts/plot_cluster2x2.ipynb
```
