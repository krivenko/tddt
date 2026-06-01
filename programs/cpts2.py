#####   CPT+Sig2 in DF-space ##################
import sys
import os
# Make sure the repo root is on the Python path when running the script directly.
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from itertools import product
from copy import deepcopy
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

from programs.utilities import *

########################## Reference system #######################

# Model parameters
U = 1.0
U1= 1.0 #_args.U1
Omega=0.0
eps = 0.0
Mu = 0.5 * U
Mu1= 0.5 * U1
V1= 1.0
V2= 1.2 #_args.t1
T=5

# time-mesh
t_max = 15.0 # 10.0 #20.0
n_t = 101 #21
t_mesh = MeshReTime(0, t_max, n_t)
tt_mesh = MeshProduct(t_mesh, t_mesh) # A 2D mesh as a direct product of t_mesh with itself
m_interp = MeshReTime(0, t_max, n_t)

# fermionic operators for the reference problem 
spin_names = ('up', 'dn')
fops = set(product(spin_names, [0, 1]))


# Keldysh branches 
FW = Branch.FORWARD
BW = Branch.BACKWARD
branches = (FW, BW)


# Ref Hamiltonian 
h0 = -Mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0) \
     +eps * (n('up', 1) + n('dn', 1)) \
     +V1 * (c_dag('up', 0) * c('up', 1) + c_dag('up', 1) * c('up', 0) \
     +      c_dag('dn', 0) * c('dn', 1) + c_dag('dn', 1) * c('dn', 0) )

params = {}
params['verbosity'] = 2                      # Verbosity level
params['hbar'] = 1.0                         # Planck's constant
params['hamiltonian_interpol'] = 'Trapezoid' # Trapezoid rule interpolation of H(t)
params['lanczos_min_matrix_size'] = 40       # Use LAPACK for subspaces of dim 32 and smaller

# Initial state
init_state = make_equilibrium_init_state(h0,
                                         fermion_indices=fops,
                                         boson_indices=set(),
                                         temperature=T,
                                         params={})

# Hamiltonian after quentch  
h = -Mu1 * (n('up', 0) + n('dn', 0)) + U1 * n('up', 0) * n('dn', 0) \
     +eps * (n('up', 1) + n('dn', 1)) \
     +V1 * (c_dag('up', 0) * c('up', 1) + c_dag('up', 1) * c('up', 0) \
     +      c_dag('dn', 0) * c('dn', 1) + c_dag('dn', 1) * c('dn', 0) )

# Reference System GF 2 sites! (paramagnetic: only 'up' block needed)
gf_struct_ref = [('up', 2)]
Gref = compute_keldysh_gf(gf_struct_ref,
                        init_state,
                        h,
                        t_mesh,
                        params)

print('GrefIm=',Gref['up'][FW,BW].data[0,0,0,0].imag)
print('GrefRe=',Gref['up'][FW,BW].data[0,0,0,0].real)

Gimp = KeldyshGF(mesh=tt_mesh) # GF for site 0 of the reference system

# Local impurity GF
for br1 in branches:
    for br2 in branches:
        Gimp[br1,br2].data[...] = Gref['up'][br1,br2].data[...,0,0]

# Make bath Keldysh GF of a single fermion with energy `eps` and occupation number `occup_n` 
def make_g(eps, occup_n, t_mesh):
    tt_mesh = MeshProduct(t_mesh, t_mesh)
    g_l = Gf(mesh=tt_mesh, target_shape=(1, 1))
    g_g = Gf(mesh=tt_mesh, target_shape=(1, 1))
    for time1, time2 in tt_mesh:
        g_g[time1, time2] = -1j * (1.0 - occup_n) * np.exp(-1j * eps * (time1 - time2))
        g_l[time1, time2] = -1j * (-occup_n) * np.exp(-1j * eps * (time1 - time2))
    return KeldyshGF.from_lesser_greater(g_l, g_g)
occup_n = 0.5
# Bare impurity GF for 1 bath cites 
Gbath = make_g(eps, occup_n, t_mesh)

# Hybridisation functions delta1, delta2 and CPT: delta21=delta2-delta1 
#delta1 = V1 * Gbath * V1
#delta2 = V2 * Gbath * V2
# Perturbation: Hybridisation functions delta21=delta2-delta1 
#delta = delta2 - delta1


# Hybridisation function delta=delta0-delta1 diff of 2 imp. models 
delta = KeldyshGF(mesh=tt_mesh)
for br1 in branches:
    for br2 in branches:
        for time1, time2 in MeshProduct(t_mesh, t_mesh):
            z1 = ContourPoint(br1, time1)
            z2 = ContourPoint(br2, time2)
            delta[z1,z2] = V2 * Gbath[z1,z2] * V2 - \
                           V1 * Gbath[z1,z2] * V1 

print('delIM=',delta[FW,BW].data[1,0].imag)
print('delRe=',delta[FW,BW].data[1,0].real)
print('GimpIm=',Gimp[FW,BW].data[0,0].imag)
print('GimpRe=',Gimp[FW,BW].data[0,0].real)

# Dual GF Gd0=(1/t-g)^(-1)=(1-tg)(**-1)t or Gd0-tgGd0=t
Q=delta
F=-delta @ Gimp
Gd0 = solve_vie2(F,Q)

# Transform to OLD-DF normalization: Gd=g*Gd0*g
Gd1=Gd0 @ Gimp
Gd = Gimp @ Gd1

# Dual Sigma2
# time dependent Interaction
def Udyn(t):
    return U1* np.cos(Omega * t)

Gd_T = Gd.T
Sig = KeldyshGF(mesh=tt_mesh)
#for br1 in branches:
#    for br2 in branches:
#        for time1, time2 in MeshProduct(t_mesh, t_mesh):
#            z1 = ContourPoint(br1, time1)
#            z2 = ContourPoint(br2, time2)
#            Sig[z1,z2] = U * Gd[z1,z2] * Gd_T[z1,z2] * Gd[z1,z2] * U 

#Sig = deepcopy(Gd0)
for br in product(Branch, Branch):  # Iterate over pairs (FW, FW), (FW, BW), (BW, FW), (BW, BW)
    Sig[br] = Gd[br] * Gd_T[br] * Gd[br]
    for t1, t2 in tt_mesh:
        Sig[br][t1, t2] *= Udyn(t1) * Udyn(t2)

# Return to the NEW-DF notation
Sig1=Sig @ Gimp
Sigd=Gimp @ Sig1

## CPT
#GimpS = Gimp 
## CPT*Sig2
GimpS = Gimp + Sigd
GimpS_herm = 0.5 * (GimpS + herm_conj(GimpS))
gdel = -GimpS @ delta
#G = solve_vie2(gdel, GimpS)
G = solve_vie2(gdel, GimpS_herm)

#TO USE HDF5
'''
import h5py
from programs.utilities import save_keldysh_gf_2pt
with h5py.File(_args.output, 'a') as f:
    # dual/sigma_2 - delete if exists, then save
    dual_grp = f.require_group('dual')
    if 'sigma_2' in dual_grp:
        del dual_grp['sigma_2']
    save_keldysh_gf_2pt(dual_grp, 'sigma_2', Sigd)
    
    # sys/G_CPT_sigma2 - delete if exists, then save
    sys_grp = f.require_group('sys')
    if 'G_CPT_sigma2' in sys_grp:
        del sys_grp['G_CPT_sigma2']
    save_keldysh_gf_2pt(sys_grp, 'G_CPT_sigma2', G)
'''

#SIMPLE SAVE TO FILE
brf, brb = Branch.FORWARD, Branch.BACKWARD
print_GF('programs/data/G_CPT_Sigma2.dat', G, brf, brb, t_mesh, idx0=0)
print_GF('programs/data/Sigma_Perturb2.dat', Sigd, brf, brb, t_mesh, idx0=0)

'''
with open('programs/data/tddt_CPTs2.dat', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        #file.write("{} {} {} {} {} \n".format(t.value,GimpS[FW,BW].data[0,t.index].real, GimpS[FW,BW].data[0,t.index].imag , Gimp[FW,BW].data[0,t.index].real, Gimp[FW,BW].data[0,t.index].imag))
        file.write("{} {} {} {} {} \n".format(t.value,G[FW,BW].data[0,t.index].real, G[FW,BW].data[0,t.index].imag , Gimp[FW,BW].data[0,t.index].real, Gimp[FW,BW].data[0,t.index].imag))
with open('programs/data/Sig2.dat', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        #file.write("{} {} {} {} {} \n".format(t.value, Sig[FW,BW].data[0,t.index].real, Sig[FW,BW].data[0,t.index].imag , Gimp[FW,BW].data[0,t.index].real, Gimp[FW,BW].data[0,t.index].imag))
        #file.write("{} {} {} {} {} \n".format(t.value, delta[FW,BW].data[0,t.index].real, delta[FW,BW].data[0,t.index].imag , Gd0[FW,BW].data[0,t.index].real, Gd0[FW,BW].data[0,t.index].imag))
        file.write("{} {} {} {} {} \n".format(t.value,Sigd[FW,BW].data[0,t.index].real, Sigd[FW,BW].data[0,t.index].imag , Gd0[FW,BW].data[0,t.index].real, Gd0[FW,BW].data[0,t.index].imag))
'''