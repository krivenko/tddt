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

import os
import numpy as np

import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, MeshBrZone, MeshCycLat, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone

from realevol.tinterp import TInterp as ti

from tddt.keldysh import Branch
from tddt.dtrilex import DualTRILEX, Channel

from tddt.lattice import local_part, lattice_fourier, SpacialArgs
from tddt.models import FiniteCluster

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

############################ Time meshes #######################################

t_max = 10.0
n_t = 51
# Time mesh for correlation functions
t_mesh = MeshReTime(0, t_max, n_t)
# Time mesh used to construct interpolators
ti_mesh = MeshReTime(0, t_max, 5001)

########################## Lattice problem #####################################

n_k = 2             # Number of k-points along each dimension
t_nn = 1.0          # Nearest neighbor hopping
t_nnn = 0.0         # Next nearest neighbor hopping

# Vector potential: Amplitude and frequency
A = 0.0
Omega = 4.0

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

U = 6.0             # Hubbard interaction at t=0
U1 = 4.0            # Hubbard interaction at t>0
V = 0.2             # Non-local interaction strength
Uch = U1 / 2
Usp = -U1 / 2

U_tq = Gf(
    mesh=MeshProduct(t_mesh, bz_mesh),
    target_shape=(2, 1, 1, 1, 1)  # 2 channels, 1 orbital
)

for t, q in U_tq.mesh:
    V_q = 2 * V * (np.cos(k[0]) + np.cos(k[1]))
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

# Energy levels of uncorrelated plaquette sites are +exx, -exx, ex
ex = 0.0
exx = 3.0

# Hopping amplitudes between site 0 and the uncorrelated sites are txx, txx, tx
tx = 0.7 * t_nn
txx = 1.0 * t_nn

# Time-dependent vector potential (x- and y-component)
Ax_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])
Ay_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])

# Temperature
T = 0.01

# Reference system
model_ref = FiniteCluster(
    # 2x2 plaquette
    [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
    # Hopping matrix
    hopping=[[-mu, txx, txx,  tx],                            # noqa: E202, E241
             [txx, exx, 0,    0 ],                            # noqa: E202, E241
             [txx, 0,   -exx, 0 ],                            # noqa: E202, E241
             [tx,  0,   0,    ex]],                           # noqa: E202, E241
    # Local interaction
    local_int=[U, 0, 0, 0],
    # Vector potential: Components along the two Cartesian axes
    vector_potential=(Ax_t, Ay_t, 0)
)

theory = DualTRILEX(model_ref, t_mesh,
                    imp_states_up=[('up', 0)],
                    imp_states_dn=[('dn', 0)])

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

###################### Construct a hybridization function ######################

# Hybridization of impurity site 0 with bath sites 1, 2, 3
print("Computing the hybridization function...")
Delta = model_ref.hybridization(theory.t_mesh, [0], [1, 2, 3], T=T)

########################### Solve D-TRILEX equations ###########################

print("Computing bare dual lines and the vertex...")
theory.compute_bare_lines_vertex(eps_tk, Delta, U_tq, U_dc)

print("Computing dual diagrams...")
theory.compute_diagrams()

print("Computing lattice Green's functions...")
g_tk = theory.compute_lattice_gf()
g_tr = lattice_fourier(g_tk, apply_to=SpacialArgs.BRZONE)

g_cpt_tk = theory.compute_lattice_gf_cpt()
g_cpt_tr = lattice_fourier(g_cpt_tk, apply_to=SpacialArgs.BRZONE)

############################# g_imp @ Gd0 @ g_imp ##############################

gd0_full_tk = theory.g_imp @ theory.Gd0_reg_tk + theory.g_imp @ theory.eps_tk
gd0_full_tk = gd0_full_tk @ theory.g_imp

gd0_full_tr = lattice_fourier(gd0_full_tk, apply_to=SpacialArgs.BRZONE)

######################### Write results to text files ##########################

# Create the data directory
os.makedirs("data", exist_ok=True)


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
