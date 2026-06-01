# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# Copyright (C) 2021-2026, I. Krivenko, V. Harkov, V. Valmispild
#
# tddt is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# tddt is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# tddt. If not, see <http://www.gnu.org/licenses/>.
#
# ##############################################################################

import argparse
import shlex
import sys
import os
import time

# Add repo root to sys.path so that `tddt` and `scripts` are importable
# regardless of where the script is invoked from.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, MeshBrZone, MeshCycLat, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone

from realevol.tinterp import TInterp as ti

from tddt.dtrilex import DualTRILEX, Channel

from tddt.lattice import local_part, lattice_fourier, SpacialArgs
from tddt.models import FiniteCluster

from scripts.utilities import write_keldysh_gf_file, save_keldysh_gf_2pt_h5

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# -----------------------------------------------------------------------
# MPI setup
# -----------------------------------------------------------------------
_has_mpi = False
_comm    = None
_rank    = 0

try:
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    _has_mpi = True
    _comm    = MPI.COMM_WORLD
    _rank    = _comm.Get_rank()
except Exception:
    pass


def get_time():
    if _has_mpi:
        return MPI.Wtime()
    return time.perf_counter()


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="D-TRILEX single-shot driver for the 2×2 plaquette cluster"
)
_parser.add_argument("--t_max",       type=float, default=10.0)
_parser.add_argument("--n_t",         type=int,   default=51)
_parser.add_argument("--n_ti",        type=int,   default=5001,
                     help="Fine interpolation mesh size")
_parser.add_argument("--n_k",         type=int,   default=2)
_parser.add_argument("--t_nn",        type=float, default=1.0)
_parser.add_argument("--t_nnn",       type=float, default=0.0)
_parser.add_argument("--A",           type=float, default=0.0)
_parser.add_argument("--Omega",       type=float, default=4.0)
_parser.add_argument("--U",           type=float, default=6.0)
_parser.add_argument("--U1",          type=float, default=4.0)
_parser.add_argument("--V",           type=float, default=0.2)
_parser.add_argument("--T",           type=float, default=0.01)
_parser.add_argument("--ex",          type=float, default=0.0)
_parser.add_argument("--exx",         type=float, default=3.0)
_parser.add_argument("--tx",          type=float, default=0.7)
_parser.add_argument("--txx",         type=float, default=1.0)
_parser.add_argument("--output_dir",  type=str,   default="data")
_parser.add_argument("--output_name", type=str,   default="cluster2x2")
_parser.add_argument("--simple_output", action="store_true",
                     help="Write plain-text files instead of HDF5")

# MPI-safe: rank 0 reads param file (optional first positional arg), broadcasts
if _rank == 0:
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        with open(sys.argv[1]) as f:
            lines = []
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    lines.append(line)
        _arg_list = shlex.split(" ".join(lines))
    else:
        _arg_list = sys.argv[1:]
    _args = _parser.parse_args(_arg_list)
else:
    _args = None

if _has_mpi:
    _args = _comm.bcast(_args, root=0)

t_max       = _args.t_max
n_t         = _args.n_t
n_ti        = _args.n_ti
n_k         = _args.n_k
t_nn        = _args.t_nn
t_nnn       = _args.t_nnn
A           = _args.A
Omega       = _args.Omega
U           = _args.U
U1          = _args.U1
V           = _args.V
T           = _args.T
ex          = _args.ex
exx         = _args.exx
tx          = _args.tx
txx         = _args.txx
output_dir  = _args.output_dir
output_name = _args.output_name

############################ Time meshes #######################################

# Time mesh for correlation functions
t_mesh = MeshReTime(0, t_max, n_t)
# Time mesh used to construct interpolators
ti_mesh = MeshReTime(0, t_max, n_ti)

########################## Lattice problem #####################################

lat = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # 2D square lattice
bz_mesh = MeshBrZone(BrillouinZone(lat), n_k)       # k-mesh on 1BZ; 0 - 2pi

# Lattice dispersion \eps_{\sigma,l,\sigma',l'}(t, k)
eps_tk = Gf(
    mesh=MeshProduct(t_mesh, bz_mesh),
    target_shape=(2, 1, 2, 1)  # 2 spins, 1 orbital
)

for t, k in eps_tk.mesh:
    A_t = A * np.cos(Omega * t.value)
    for spin in range(2):
        eps_tk[t, k][spin, 0, spin, 0] = \
            2 * t_nn * (np.cos(k[0] - A_t) + np.cos(k[1] - A_t)) \
            + 4 * t_nnn * np.cos(2 * (k[0] - A_t)) * np.cos(2 * (k[1] - A_t))

# Interaction U^\varsigma_{l_1,l_2,l_3,l_4}(t, q)

Uch = U1 / 2
Usp = -U1 / 2

U_tq = Gf(
    mesh=MeshProduct(t_mesh, bz_mesh),
    target_shape=(2, 1, 1, 1, 1)  # 2 channels, 1 orbital
)

for t, q in U_tq.mesh:
    V_q = 2 * V * (np.cos(q[0]) + np.cos(q[1]))
    U_tq[t, q][Channel.CHARGE.value] = Uch + V_q
    U_tq[t, q][Channel.SPIN.value] = Usp + V_q

# Double-counting interaction shifts
U_dc = np.zeros((2, 1, 1, 1, 1))  # 2 channels, 1 orbital
U_dc[Channel.CHARGE.value] = Uch
U_dc[Channel.SPIN.value] = Usp

########################## Reference system ####################################

# Correlated plaquette site (0)
mu = 0.5 * U        # Chemical potential at t=0
mu1 = 0.5 * U1      # Chemical potential at t>0

# Time-dependent vector potential (x- and y-component)
Ax_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])
Ay_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])

# Reference system
model_ref = FiniteCluster(
    # 2x2 plaquette
    [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
    # Hopping matrix
    hopping=[[-mu,  txx,  txx,  tx],                            # noqa: E202, E241
             [ txx,  exx,  0,    0 ],                           # noqa: E202, E241
             [ txx,  0,   -exx,  0 ],                           # noqa: E202, E241
             [ tx,   0,    0,    ex]],                          # noqa: E202, E241
    # Local interaction
    local_int=[U, 0, 0, 0],
    # Vector potential: Components along the two Cartesian axes
    vector_potential=(Ax_t, Ay_t, 0)
)

theory = DualTRILEX(model_ref, t_mesh,
                    imp_states_up=[('up', 0)],
                    imp_states_dn=[('dn', 0)])

t_start = get_time()

# Compute initial thermal state of the reference system
theory.compute_ref_init_state(T, verbosity=1)

# Set local Hubbard interaction and energy level on site 0 after quench
model_ref.local_int[0] = U1
model_ref.hopping[0, 0] = -mu1

# Compute correlation functions of the reference system
print("Computing correlators of the reference system...")
theory.compute_ref_correlators(verbosity=2,
                               hamiltonian_interpol='Trapezoid',
                               lanczos_min_matrix_size=40)
t_ref_done = get_time() - t_start
print(f"Reference correlators done  [Time: {t_ref_done:.3f} s]")

###################### Construct a hybridization function ######################

# Hybridization of impurity site 0 with bath sites 1, 2, 3
print("Computing the hybridization function...")
Delta = model_ref.hybridization(theory.t_mesh, [0], [1, 2, 3], T=T)

########################### Solve D-TRILEX equations ###########################

print("Computing bare dual lines and the vertex...")
t_bare_start = get_time()
theory.compute_bare_lines_vertex(eps_tk, Delta, U_tq, U_dc)
t_bare = get_time() - t_bare_start
print(f"Bare lines done  [Time: {t_bare:.3f} s]")

print("Computing dual diagrams...")
t_diag_start = get_time()
theory.compute_diagrams()
t_diag = get_time() - t_diag_start
print(f"Diagrams done  [Time: {t_diag:.3f} s]")

print("Computing lattice Green's functions...")
t_lat_start = get_time()
g_tk = theory.compute_lattice_gf()
g_tr = lattice_fourier(g_tk, apply_to=SpacialArgs.BRZONE)

g_cpt_tk = theory.compute_lattice_gf_cpt()
g_cpt_tr = lattice_fourier(g_cpt_tk, apply_to=SpacialArgs.BRZONE)

############################# g_imp @ Gd0 @ g_imp ##############################

gd0_full_tk = theory.g_imp @ theory.Gd0_reg_tk + theory.g_imp @ theory.eps_tk
gd0_full_tk = gd0_full_tk @ theory.g_imp

gd0_full_tr = lattice_fourier(gd0_full_tk, apply_to=SpacialArgs.BRZONE)

t_lat = get_time() - t_lat_start
t_total = get_time() - t_start
print(f"Lattice GFs done  [Time: {t_lat:.3f} s]")
print()
print("=== TIMING SUMMARY ===")
print(f"Reference correlators:  {t_ref_done:.3f} s")
print(f"Bare lines + vertex:    {t_bare:.3f} s")
print(f"Dual diagrams:          {t_diag:.3f} s")
print(f"Lattice GFs:            {t_lat:.3f} s")
print(f"TOTAL TIME:             {t_total:.3f} s")
print("======================")

######################### Write results ########################################

def write_keldysh_gf_file(filename, g, spc_point=None, target_indices=()):
    """
    Write (FW, FW) and (FW, BW) components of a 2-point KeldyshGF
    object to a text file.
    """
    FW, BW = Branch
    t_mesh = g.time_mesh.components[1]
    t0 = next(iter(t_mesh))
    with open(filename, 'w') as file:
        file.write("# t Re (FW, BW) Im (FW, BW) Re (BW, FW) Im (BW, FW)\n")
        for t in t_mesh:
            mesh_point = (t, t0) if (spc_point is None) else (t, t0, spc_point)
            col_data = (t.value,
                        g[FW, BW][*mesh_point][*target_indices].real,
                        g[FW, BW][*mesh_point][*target_indices].imag,
                        g[BW, FW][*mesh_point][*target_indices].real,
                        g[BW, FW][*mesh_point][*target_indices].imag)
            file.write("{} {} {} {}\n".format(*col_data))


write_keldysh_gf_file('data/tddt_ref_sys_t0_00.txt',
                      theory.g_ref, target_indices=(0, 0, 0, 0))
write_keldysh_gf_file('data/tddt_ref_sys_t0_01.txt',
                      theory.g_ref, target_indices=(0, 0, 0, 1))

r_mesh = MeshCycLat(lat, n_k)
r0, r1 = list(r_mesh)[:2]
k0, k1 = list(bz_mesh)[:2]
write_keldysh_gf_file("data/tddt_Gd0_R_k0.txt",
                      gd0_full_tr, spc_point=r0, target_indices=(0, 0, 0, 0))
write_keldysh_gf_file("data/tddt_Gd0_R_k1.txt",
                      gd0_full_tr, spc_point=r1, target_indices=(0, 0, 0, 0))

write_keldysh_gf_file("data/tddt_t0_loc.txt",
                      local_part(g_tk), target_indices=(0, 0, 0, 0))
write_keldysh_gf_file("data/tddt_CPT.txt",
                      local_part(g_cpt_tk), target_indices=(0, 0, 0, 0))
write_keldysh_gf_file("data/sigma_dual.txt",
                      local_part(theory.Sigma_tk), target_indices=(0, 0, 0, 0))
write_keldysh_gf_file("data/tddt_01.txt",
                      g_tr, spc_point=r1, target_indices=(0, 0, 0, 0))
write_keldysh_gf_file("data/tddt_CPT_01.txt",
                      g_cpt_tr, spc_point=r1, target_indices=(0, 0, 0, 0))

for ki, k in enumerate(bz_mesh):
    write_keldysh_gf_file(f"data/tddt_G_lat_CPT_k{ki}.txt",
                          g_cpt_tk, spc_point=k, target_indices=(0, 0, 0, 0))
    write_keldysh_gf_file(f"data/tddt_G_lat_k{ki}.txt",
                          g_tk, spc_point=k, target_indices=(0, 0, 0, 0))
