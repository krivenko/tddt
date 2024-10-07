import unittest
import pytest
from itertools import product
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import expm  # Matrix exponential

import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, Gf
from triqs.gf import MeshProduct # A direct product of 1D meshes

from realevol.tinterp import TInterp as ti
from realevol.operators_tinterp import c, c_dag, n
from realevol.init_state import make_equilibrium_init_state
from realevol.realevol import compute_expectval

#from tddt.keldysh import Branch, KeldyshGF, from_lesser_greater, conv
from tddt.keldysh import Branch, KeldyshGF, conv
from tddt.realevol import (
    compute_keldysh_gf,
    compute_keldysh_gf_element,
    compute_keldysh_correlator_2t,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)
from tddt.diagrams import (polarization_2nd_order,
                           selfenergy_2nd_order,
                           selfenergy_2nd_order_hf)

#from tddt.keldysh import Branch, lesser, greater, retarded, advanced
from tddt.keldysh import ContourPoint
from tddt.keldysh import herm_conj
from tddt.keldysh import Singular2PKeldyshGF

# Import some TRIQS modules related to lattice
from triqs.lattice import BravaisLattice, BrillouinZone
from triqs.gf import MeshBrillouinZone

from tddt.vie2 import solve_vie2
import matplotlib.pyplot as plt
from h5 import HDFArchive
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

########################## Reference system #######################

# Model parameters
U = 1.0
Uch = U/2
Usp = -U/2
mu = 0.5 * U
#eps = 0.2
t1 = 0.1 # nearest neighbor hopping 
t2 = 0.0 # next nearest neighbor hopping
A = 0.0
Omega = 10
T = 0.0

# time-mesh
t_max = 8.0 # 10.0 #20.0
n_t = 31 #11 #21
t_mesh = MeshReTime(0, t_max, n_t)
tt_mesh = MeshProduct(t_mesh, t_mesh) # A 2D mesh as a direct product of t_mesh with itself
ttt_mesh = MeshProduct(t_mesh, t_mesh, t_mesh)
m_interp = MeshReTime(0, t_max, n_t)

#k-mesh
lat = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # 2D square lattice
bz = BrillouinZone(lat)  # Brillouin zone of the lattice
n_k = 2 # Number of k-points along each dimension
bz_mesh = MeshBrillouinZone(bz, n_k) # k-mesh on 1BZ; 0 - 2pi
nkx = n_k
nky = n_k
nkz = 1

#mixed-mesh
tk_mesh = MeshProduct(t_mesh, bz_mesh)
ttk_mesh = MeshProduct(t_mesh, t_mesh, bz_mesh)
tttk_mesh = MeshProduct(t_mesh, t_mesh, t_mesh, bz_mesh)


# Keldysh branches 
FW = Branch.FORWARD
BW = Branch.BACKWARD
branches = (FW, BW)
for k in bz_mesh:
    print(k)

# time dependent hopping for Hamiltonian
dt_pos = ti(m_interp, np.array([t1*np.exp(1.j * A * np.cos(Omega * x)) for x in m_interp]))
dt_neg = ti(m_interp, np.array([t1*np.exp(-1.j * A * np.cos(Omega * x)) for x in m_interp]))
print(dt_pos.data)
print(dt_pos.data.shape)
#A_x = 0.1
#dt_pos_x = ti(m_interp, np.array([t1*np.exp(1.j * A_x * np.cos(Omega * x)) for x in m_interp]))
#dt_neg_x = ti(m_interp, np.array([t1*np.exp(-1.j * A_x * np.cos(Omega * x)) for x in m_interp]))


# Dispersion reference system 
#TODO: compare to dt_pos/ dt_neg
def V(axis, sign, t):
    """ axis = x,y
        sign = -1,+1
    """
    V = t1 * np.exp(sign * 1.j * A * np.cos(Omega * t))
    return V
for i in t_mesh:
    print(V('x', +1, i))


# fermionic operators for the reference problem 
spin_names = ('up', 'dn')
fops = set(product(spin_names, [0, 1, 2, 3]))

# Initial Hamiltonian #TODO: check which parameters to add 
h0 = -mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0) \
     -mu * (n('up', 1) + n('dn', 1)) + U * n('up', 1) * n('dn', 1) \
     -mu * (n('up', 2) + n('dn', 2)) + U * n('up', 2) * n('dn', 2) \
     -mu * (n('up', 3) + n('dn', 3)) + U * n('up', 3) * n('dn', 3)

h0 = h0 + \
    sum((dt_pos + dt_neg) * c_dag(sn, 0) * c(sn, 1) + (dt_pos + dt_neg) * c_dag(sn, 1) * c(sn, 0)
        for sn in spin_names) + \
    sum((dt_pos + dt_neg) * c_dag(sn, 0) * c(sn, 2) + (dt_pos + dt_neg) * c_dag(sn, 2) * c(sn, 0)
        for sn in spin_names) + \
    sum((dt_pos + dt_neg) * c_dag(sn, 1) * c(sn, 3) + (dt_pos + dt_neg) * c_dag(sn, 3) * c(sn, 1)
        for sn in spin_names) + \
    sum((dt_pos + dt_neg) * c_dag(sn, 2) * c(sn, 3) + (dt_pos + dt_neg) * c_dag(sn, 3) * c(sn, 2)
        for sn in spin_names)

#h0 = h0 + \
#    sum((dt_pos) * c_dag(sn, 0) * c(sn, 1) + (dt_neg) * c_dag(sn, 1) * c(sn, 0)
#        for sn in spin_names) + \
#    sum((dt_neg) * c_dag(sn, 0) * c(sn, 2) + (dt_pos) * c_dag(sn, 2) * c(sn, 0)
#        for sn in spin_names) + \
#    sum((dt_pos) * c_dag(sn, 1) * c(sn, 3) + (dt_neg) * c_dag(sn, 3) * c(sn, 1)
#        for sn in spin_names) + \
#    sum((dt_neg) * c_dag(sn, 2) * c(sn, 3) + (dt_pos) * c_dag(sn, 3) * c(sn, 2)
#        for sn in spin_names)

init_state = make_equilibrium_init_state(h0,
                                         fermion_indices=fops,
                                         boson_indices=set(),
                                         temperature=T,
                                         params={})


# Hamiltonian after quench
h = h0 + \
    sum((dt_pos + dt_neg) * c_dag(sn, 0) * c(sn, 1) + (dt_pos + dt_neg) * c_dag(sn, 1) * c(sn, 0)
        for sn in spin_names) + \
    sum((dt_pos + dt_neg) * c_dag(sn, 0) * c(sn, 2) + (dt_pos + dt_neg) * c_dag(sn, 2) * c(sn, 0)
        for sn in spin_names) + \
    sum((dt_pos + dt_neg) * c_dag(sn, 1) * c(sn, 3) + (dt_pos + dt_neg) * c_dag(sn, 3) * c(sn, 1)
        for sn in spin_names) + \
    sum((dt_pos + dt_neg) * c_dag(sn, 2) * c(sn, 3) + (dt_pos + dt_neg) * c_dag(sn, 3) * c(sn, 2)
        for sn in spin_names)

#h = h0 + \
#    sum((dt_pos) * c_dag(sn, 0) * c(sn, 1) + (dt_neg) * c_dag(sn, 1) * c(sn, 0)
#        for sn in spin_names) + \
#    sum((dt_neg) * c_dag(sn, 0) * c(sn, 2) + (dt_pos) * c_dag(sn, 2) * c(sn, 0)
#        for sn in spin_names) + \
#    sum((dt_pos) * c_dag(sn, 1) * c(sn, 3) + (dt_neg) * c_dag(sn, 3) * c(sn, 1)
#        for sn in spin_names) + \
#    sum((dt_neg) * c_dag(sn, 2) * c(sn, 3) + (dt_pos) * c_dag(sn, 3) * c(sn, 2)
#        for sn in spin_names)

h = h0

params = {}
params['verbosity'] = 2                      # Verbosity level
params['hbar'] = 1.0                         # Planck's constant
params['hamiltonian_interpol'] = 'Trapezoid' # Trapezoid rule interpolation of H(t)
params['lanczos_min_matrix_size'] = 40       # Use LAPACK for subspaces of dim 32 and smaller


# Reference System GF
gf_struct_ref = [('up', 4), ('dn', 4)]
gf_ref = compute_keldysh_gf(gf_struct_ref,
                        init_state,
                        h,
                        t_mesh,
                        params)


gref = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
print(gref[FW,FW].data.shape)
#exit()
#print(gimp[FW,FW][0,0].data.shape)
#print(gf['up'][FW,FW].data.shape)
#print(gf)
#print(gf['up'])
#print(gf['up'][FW,FW])
#print(dir(gf))
#print(dir(gf['up']))
#rint(dir(gf['up'][FW,FW]))
print(gf_ref['up'][FW,FW].data.shape)


for br1 in branches:
    for br2 in branches:
        for i in range(4): #k-points
            gref[br1,br2].data[...,i,0,0] = gf_ref['up'][br1,br2].data[...,i,i]
            gref[br1,br2].data[...,i,1,1] = gf_ref['dn'][br1,br2].data[...,i,i]
"""
gref_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                for site1 in range(4): # spin up and dn
                    for site2 in range(4): # spin up and dn
                        if abs(site1 - site2) == 0:
                            dx = 0.0
                            dy = 0.0
                        elif abs(site1 - site2) == 1:
                            if site1 == 1 & site2 == 2:
                                dx = -1.0
                                dy = 1.0 
                            elif site1 == 2 & site2 == 1:
                                dx = 1.0
                                dy = -1.0
                            else:
                                dx = site2 - site1
                                dy = 0.0
                        elif abs(site1 - site2) == 2:
                            dx = 0.0
                            dy = 0.5*(site2 - site1)
                        else:
                            dx = (site2 - site1)/3
                            dy = (site2 - site1)/3
                        gref_K[br1,br2].data[...,0,0,0] += gf_ref['up'][br1,br2].data[...,site1,site2]
                        gref_K[br1,br2].data[...,0,1,1] += gf_ref['dn'][br1,br2].data[...,site1,site2]

                        gref_K[br1,br2].data[...,1,0,0] += gf_ref['up'][br1,br2].data[...,site1,site2] * np.exp(-1j * np.pi * dy)
                        gref_K[br1,br2].data[...,1,1,1] += gf_ref['dn'][br1,br2].data[...,site1,site2]* np.exp(-1j * np.pi * dy)

                        gref_K[br1,br2].data[...,2,0,0] += gf_ref['up'][br1,br2].data[...,site1,site2] * np.exp(-1j * np.pi * dx)
                        gref_K[br1,br2].data[...,2,1,1] += gf_ref['dn'][br1,br2].data[...,site1,site2]* np.exp(-1j * np.pi * dx) 

                        gref_K[br1,br2].data[...,3,0,0] += gf_ref['up'][br1,br2].data[...,site1,site2] * np.exp(-1j * np.pi * dx - 1j * np.pi * dy)
                        gref_K[br1,br2].data[...,3,1,1] += gf_ref['dn'][br1,br2].data[...,site1,site2]* np.exp(-1j * np.pi * dx - 1j * np.pi * dy) 

print(gref_K[FW,FW].data[0,:,0,0,0])
"""
gref_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for site1 in range(4): # spin up and dn
                for site2 in range(4): # spin up and dn
                    if abs(site1 - site2) == 0:
                        dx = 0.0
                        dy = 0.0
                    elif abs(site1 - site2) == 1:
                        if site1 == 1 & site2 == 2:
                            dx = -1.0
                            dy = 1.0 
                        elif site1 == 2 & site2 == 1:
                            dx = 1.0
                            dy = -1.0
                        else:
                            dx = site2 - site1
                            dy = 0.0
                    elif abs(site1 - site2) == 2:
                        dx = 0.0
                        dy = 0.5*(site2 - site1)
                    else:
                        dx = (site2 - site1)/3
                        dy = (site2 - site1)/3
                    gref_K[br1,br2].data[time1.index,time2.index,0,0,0] += gf_ref['up'][br1,br2].data[time1.index,time2.index,site1,site2]
                    gref_K[br1,br2].data[time1.index,time2.index,0,1,1] += gf_ref['dn'][br1,br2].data[time1.index,time2.index,site1,site2]
                    gref_K[br1,br2].data[time1.index,time2.index,1,0,0] += gf_ref['up'][br1,br2].data[time1.index,time2.index,site1,site2] * np.exp(-1j * np.pi * dy)
                    gref_K[br1,br2].data[time1.index,time2.index,1,1,1] += gf_ref['dn'][br1,br2].data[time1.index,time2.index,site1,site2]* np.exp(-1j * np.pi * dy)
                    gref_K[br1,br2].data[time1.index,time2.index,2,0,0] += gf_ref['up'][br1,br2].data[time1.index,time2.index,site1,site2] * np.exp(-1j * np.pi * dx)
                    gref_K[br1,br2].data[time1.index,time2.index,2,1,1] += gf_ref['dn'][br1,br2].data[time1.index,time2.index,site1,site2]* np.exp(-1j * np.pi * dx) 
                    gref_K[br1,br2].data[time1.index,time2.index,3,0,0] += gf_ref['up'][br1,br2].data[time1.index,time2.index,site1,site2] * np.exp(-1j * np.pi * dx - 1j * np.pi * dy)
                    gref_K[br1,br2].data[time1.index,time2.index,3,1,1] += gf_ref['dn'][br1,br2].data[time1.index,time2.index,site1,site2]* np.exp(-1j * np.pi * dx - 1j * np.pi * dy) 

print(gref_K[FW,FW].data[0,:,0,0,0])

with open('data/ref_sys_T_t0_loc.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gf_ref['up'][FW,FW].data[0,t.index,0,0].real, gf_ref['up'][FW,FW].data[0,t.index,0,0].imag, 
                                          gf_ref['up'][FW,BW].data[0,t.index,0,0].real, gf_ref['up'][FW,BW].data[0,t.index,0,0].imag))

with open('data/ref_sys_01.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gf_ref['up'][FW,FW].data[0,t.index,0,1].real, gf_ref['up'][FW,FW].data[0,t.index,0,1].imag, 
                                          gf_ref['up'][FW,BW].data[0,t.index,0,1].real, gf_ref['up'][FW,BW].data[0,t.index,0,1].imag))
with open('data/ref_sys_02.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gf_ref['up'][FW,FW].data[0,t.index,0,2].real, gf_ref['up'][FW,FW].data[0,t.index,0,2].imag, 
                                          gf_ref['up'][FW,BW].data[0,t.index,0,2].real, gf_ref['up'][FW,BW].data[0,t.index,0,2].imag))
with open('data/ref_sys_03.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gf_ref['up'][FW,FW].data[0,t.index,0,3].real, gf_ref['up'][FW,FW].data[0,t.index,0,3].imag, 
                                          gf_ref['up'][FW,BW].data[0,t.index,0,3].real, gf_ref['up'][FW,BW].data[0,t.index,0,3].imag))

with open('data/ref_sys_T_t0_k0.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,FW).real (FW,FW).imag (FW,BW).real (FW,BW).imag)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gref_K[FW,FW].data[0,t.index,0,0,0].real, gref_K[FW,FW].data[0,t.index,0,0,0].imag, gref_K[FW,BW].data[0,t.index,0,0,0].real, gref_K[FW,BW].data[0,t.index,0,0,0].imag))

with open('data/ref_sys_T_t0_k1.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(gref_K[FW,FW].data[0,t.index,1,0,0].real, gref_K[FW,FW].data[0,t.index,1,0,0].imag, gref_K[FW,BW].data[0,t.index,1,0,0].real, gref_K[FW,BW].data[0,t.index,1,0,0].imag))

with open('data/ref_sys_T_t0_k2.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(gref_K[FW,FW].data[0,t.index,2,0,0].real, gref_K[FW,FW].data[0,t.index,2,0,0].imag, gref_K[FW,BW].data[0,t.index,2,0,0].real, gref_K[FW,BW].data[0,t.index,2,0,0].imag))

with open('data/ref_sys_T_t0_k3.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(gref_K[FW,FW].data[0,t.index,3,0,0].real, gref_K[FW,FW].data[0,t.index,3,0,0].imag, gref_K[FW,BW].data[0,t.index,3,0,0].real, gref_K[FW,BW].data[0,t.index,3,0,0].imag))
