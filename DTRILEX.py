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
t1 = 1.0 # nearest neighbor hopping 
t2 = 0.0 # next nearest neighbor hopping
A = 0.0
Omega = 10
T = 1.0

# time-mesh
t_max = 1.0 # 10.0 #20.0
n_t = 6 #11 #21
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


# time dependent hopping for Hamiltonian
dt_pos = ti(m_interp, np.array([t1*np.exp(1.j * A * np.cos(Omega * x)) for x in m_interp]))
dt_neg = ti(m_interp, np.array([t1*np.exp(-1.j * A * np.cos(Omega * x)) for x in m_interp]))
print(dt_pos.data)
print(dt_pos.data.shape)

# Lattice dispersion

eps_tk = Gf(mesh=tk_mesh, target_shape=(2, 2))

for t, k in tk_mesh:
    #eps_tk[t, k] = -2 * np.array([[[1, 0], [0, -1]]]) * np.cos(t.value) * (np.cos(k[0]) + np.cos(k[1]))
    eps_tk[t, k] = -2.0*t1*(np.cos(k[0]-A*np.cos(Omega * t.value))+np.cos(k[1]-A*np.cos(Omega * t.value))) \
                   -4.0*t2*np.cos(2*(k[0]-A*np.cos(Omega * t.value)))*np.cos(2*(k[1]-A*np.cos(Omega * t.value)))
    
eps_loc = np.mean(eps_tk.data, axis = 1)

for t, k in tk_mesh:
    eps_tk[t, k] = eps_tk[t, k] - eps_loc[t.index, :]

#print(eps_tk.data[0,0,:,:])

eps_s2p_K = Singular2PKeldyshGF.from_retime(eps_tk)


#for t, k in tk_mesh:
#    print('eps_s2p_K: ', eps_s2p_K[FW][t,k])
#print(eps_s2p_K.arg_index_shapes)

#print('eps_s2p_K:')
#print(eps_s2p_K[FW][t,:][0,0].data)
#print(eps_s2p_K[FW][t,:].data)

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

# Non-local interaction
def Vq(k, channel):
    Vq = 0.0
    return Vq


################################# OLD ################################
#previous dispersion defined as function. USE: eps_s2p_K
def eps_k(t1,t2,k,time):
    eps_k = -2.0*t1*(np.cos(k[0]-A*np.cos(time))+np.cos(k[1]-A*np.cos(time))) \
            -4.0*t2*np.cos(k[0]-A*np.cos(time))*np.cos(k[1]-A*np.cos(time))
    return eps_k

eps_loc = np.zeros(n_t)
eps_matrix = np.zeros([bz_mesh.dims[0], bz_mesh.dims[1], bz_mesh.dims[2], n_t])
eps_mat = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))

#print(bz_mesh.dims[0]*bz_mesh.dims[1]*bz_mesh.dims[2])
#print(dir(t_mesh))
#print('t_mesh.dims: ', t_mesh.dims)
#print('values: ', bz_mesh.values)
#print('dims: ', bz_mesh.dims)
#print('domain: ', bz_mesh.domain)
#print('indes_to_linear: ', bz_mesh.index_to_linear)
#print('units: ', bz_mesh.units)
#eps_k_matrix = np.zeros(k.) 
for time in t_mesh:
    for k in bz_mesh:
        eps_loc[time.index] += eps_k(t1,t2,k,time.value)
        eps_matrix[k.index[0],k.index[1],k.index[2],time.index] += eps_k(t1,t2,k,time.value)
        print('k: ', k, 'time: ', time.value)
        print('k.index: ', k.index, 'time.index: ', time.index)
        print(eps_k(t1,t2,k,time.value))
        #print(time.value)
        #print(np.cos(time.value))

eps_loc = eps_loc/(bz_mesh.dims[0]*bz_mesh.dims[1]*bz_mesh.dims[2])
print('eps_matrix: ', eps_matrix)

print('eps_loc: ', eps_loc)
################################# OLD END ################################

# Bare lattice interaction 
Uq = Gf(mesh=tk_mesh, target_shape=(2, 2))
Uq_tilde = Gf(mesh=tk_mesh, target_shape=(2, 2))

for t, k in tk_mesh:
    #eps_tk[t, k] = -2 * np.array([[[1, 0], [0, -1]]]) * np.cos(t.value) * (np.cos(k[0]) + np.cos(k[1]))
    Uq[t, k][0,0] = Uch + Vq(k, 0) # charge
    Uq[t, k][1,1] = Usp + Vq(k, 1) # spin
    Uq_tilde[t, k][0,0] = Uq[t, k][0,0] - 0.5*Uch
    Uq_tilde[t, k][1,1] = Uq[t, k][1,1] - 0.5*Usp

Uq_s2p_K = Singular2PKeldyshGF.from_retime(Uq)
Uq_tilde_s2p_K = Singular2PKeldyshGF.from_retime(Uq_tilde)

print(Uq_tilde_s2p_K[FW].data)

for t, k in tk_mesh:
    print('Uq_sp2_K: ', Uq_s2p_K[FW][t,k][0,0])

################################# OLD Uq ################################
Uq = np.zeros([bz_mesh.dims[0], bz_mesh.dims[1], bz_mesh.dims[2], 2])

for k in bz_mesh:
    for ch in range(2):
        Uq[k.index[0], k.index[1], k.index[2], ch] += Uch + Vq(k, ch) if ch == 0 else Usp + Vq(k, ch)
################################# OLD Uq END ################################


# FFT on dispersion
#TODO: check ifftn vs fftn -> which one to use?
#eps_R = np.empty_like(eps_matrix, dtype = 'complex')

#for time in t_mesh:
#    eps_K = eps_matrix[:,:,:,time.index]
#    eps_R[:,:,:,time.index] = np.fft.ifftn(eps_K, axes=(0,1,2))
#print(eps_R[:,:,0,0])

eps_R = np.empty_like(eps_matrix, dtype = 'complex')
eps_R_test = np.empty_like(eps_matrix, dtype = 'complex')
for time in t_mesh:
    #print(eps_s2p_K[FW].data.shape)
    eps_R[:,:,:,time.index]
    #print(eps_s2p_K[FW][time,:][0,0].data.reshape(3,3,1))
    eps_R_test[:,:,:,time.index] = np.fft.ifftn(eps_s2p_K[FW][time,:][0,0].data.reshape(nkx,nky,nkz), axes=(0,1,2))
    #print(eps_R[FW].data[0,:,:,0,0])
    ##print(eps_s2p_K[FW].data[time.linear_index, :, 0,0])
    #print(time.value)

print('################')
for time in t_mesh:
    eps_K = eps_matrix[:,:,:,time.index]
    eps_R[:,:,:,time.index] = np.fft.ifftn(eps_K, axes=(0,1))
    print(time)
    print('eps_R', eps_R[...,time.index])
    print('eps_R_test', eps_R_test[...,time.index])

for k in bz_mesh:
    print(k.index[0], k.index[1], k.index[2])
    print(eps_R[k.index[0], k.index[1], k.index[2],0])



############################ FFT OLD ###################################

#from convolution import convolution_fft

def from_Kplusq_to_R(g: KeldyshGF):
    # PROBLEMS: How to impliment dynamecly so there is a loop over different indices like spin, orb etc.
    #           FFT depends on k dependency of function we are considering e.g. k+q or just k
    #           The way Matteo implemented it only works if k mesh starts from Gamma! 
    assert g.n_args == 2, "g must be a 2-point GF"

    #gR = KeldyshGF(mesh=g.mesh, arg_index_shapes=g.arg_index_shapes)
    gR = g
    #Gr = np.fft.ifftn(funct1, axes=axis)
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                    for time1, time2 in MeshProduct(t_mesh, t_mesh):
                        #gR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = \
                        #               np.fft.ifftn(g[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm])
                        g_reshaped = g[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm].reshape(nkx,nky,nkz)
                        g_fft = np.fft.ifftn(g_reshaped, axes=(0,1))
                        g_fft = g_fft.reshape(nkx*nky*nkz,)
                        gR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = g_fft
    return gR

def from_K_to_R(g: KeldyshGF):
    # PROBLEMS: How to impliment dynamecly so there is a loop over different indices like spin, orb etc.
    #           FFT depends on k dependency of function we are considering e.g. k+q or just k
    #           The way Matteo implemented it only works if k mesh starts from Gamma! 
    assert g.n_args == 2, "g must be a 2-point GF"

    #gR = KeldyshGF(mesh=g.mesh, arg_index_shapes=g.arg_index_shapes)
    gR = g
    #Gr = np.fft.ifftn(funct1, axes=axis)
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                    for time1, time2 in MeshProduct(t_mesh, t_mesh):
                        #gR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = \
                        #               np.fft.ifftn(g[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm])
                        g_reshaped = g[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm].reshape(nkx,nky,nkz)
                        g_fft = np.fft.fftn(g_reshaped, axes=(0,1))
                        g_fft = g_fft.reshape(nkx*nky*nkz,)
                        gR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = g_fft
    return gR

#Gd0R = from_Kplusq_to_R(Gd0_test)
#Gd0R = from_K_to_R(Gd0_test)
#print(Gd0_test[FW,FW].data.shape)

############################ FFT OLD END ###################################


# fermionic operators for the reference problem 
spin_names = ('up', 'dn')
fops = set(product(spin_names, [0, 1, 2, 3, 4]))
#print(fops)


# Initial Hamiltonian #TODO: check which parameters to add 
h0 = -mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0) \
     -mu * (n('up', 1) + n('dn', 1)) \
     -mu * (n('up', 2) + n('dn', 2)) \
     -mu * (n('up', 3) + n('dn', 3)) \
     -mu * (n('up', 4) + n('dn', 4)) \


init_state = make_equilibrium_init_state(h0,
                                         fermion_indices=fops,
                                         boson_indices=set(),
                                         temperature=T,
                                         params={})


# Hamiltonian after quench
h = h0 + \
    sum(dt_pos * c_dag(sn, 0) * c(sn, 1) + dt_neg * c_dag(sn, 1) * c(sn, 0)
        for sn in spin_names) + \
    sum(dt_pos * c_dag(sn, 0) * c(sn, 2) + dt_neg * c_dag(sn, 2) * c(sn, 0)
        for sn in spin_names) + \
    sum(dt_neg * c_dag(sn, 0) * c(sn, 3) + dt_pos * c_dag(sn, 3) * c(sn, 0)
        for sn in spin_names) + \
    sum(dt_neg * c_dag(sn, 0) * c(sn, 4) + dt_pos * c_dag(sn, 4) * c(sn, 0)
        for sn in spin_names)

#h = h0 + \
#    sum(V('y', 1) * c_dag(sn, 0) * c(sn, 1) + V('y', -1) * c_dag(sn, 1) * c(sn, 0)
#          for sn in spin_names) + \
#    sum(V('x', 1) * c_dag(sn, 0) * c(sn, 2) + V('x', -1) * c_dag(sn, 2) * c(sn, 0)
#          for sn in spin_names) + \
#    sum(V('y', -1) * c_dag(sn, 0) * c(sn, 3) + V('y', 1) * c_dag(sn, 3) * c(sn, 0)
#          for sn in spin_names) + \
#    sum(V('x', -1) * c_dag(sn, 0) * c(sn, 4) + V('x', 1) * c_dag(sn, 4) * c(sn, 0)
#          for sn in spin_names)


params = {}
params['verbosity'] = 2                      # Verbosity level
params['hbar'] = 1.0                         # Planck's constant
params['hamiltonian_interpol'] = 'Trapezoid' # Trapezoid rule interpolation of H(t)
params['lanczos_min_matrix_size'] = 40       # Use LAPACK for subspaces of dim 32 and smaller


# Impurity GF
# TODO: remove if not needed
#gf_struct = [('up', 1), ('dn', 1)]
#gf = compute_keldysh_gf(gf_struct,
#                        init_state,
#                       h,
#                        t_mesh,
#                        params)

# Reference System GF
gf_struct_ref = [('up', 5), ('dn', 5)]
gf_ref = compute_keldysh_gf(gf_struct_ref,
                        init_state,
                        h,
                        t_mesh,
                        params)


gimp = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,))) # GF for site 0 of the reference system
gref = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,5), (2,5))) # spin, site (11, 11, 2, 5, 2, 5)
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
        gimp[br1,br2][0,0].data[...] = gf_ref['up'][br1,br2].data[...,0,0]
        gimp[br1,br2][1,1].data[...] = gf_ref['dn'][br1,br2].data[...,0,0]
        for i in range(5): # sites
            gref[br1,br2].data[...,0,i,0,i] = gf_ref['up'][br1,br2].data[...,i,i]
            gref[br1,br2].data[...,1,i,1,i] = gf_ref['dn'][br1,br2].data[...,i,i] #TODO: multiply by -i (see eq. 32) ?????????


# Hybridisation function delta
delta = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))

for sp in (0,1):
    for br1 in branches:
        for br2 in branches:
            for i in range(1,5):
                for time1, time2 in MeshProduct(t_mesh, t_mesh):
                    #print(ContourPoint(Branch.FORWARD, time1).t)
                    #print(ContourPoint(Branch.FORWARD, time1).t.value)
                    z1 = ContourPoint(br1, time1)
                    z2 = ContourPoint(br2, time2)
                    delta[z1,z2][sp,sp] = V('x',+1,time1.linear_index) * gref[z1,z2][sp,1][sp,1] * V('x',-1,time2.linear_index) \
                                          + V('y',+1,time1.linear_index) * gref[z1,z2][sp,2][sp,2]  * V('y',-1,time2.linear_index) \
                                          + V('x',-1,time1.linear_index) * gref[z1,z2][sp,3][sp,3]  * V('x',+1,time2.linear_index) \
                                          + V('y',-1,time1.linear_index) * gref[z1,z2][sp,4][sp,4]  * V('y',+1,time2.linear_index)

"""
gimp_eps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_gimp = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_gimp_new = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_gimp_eps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))


for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                #eps_gimp[br1,br2][time1,time2,k] = eps_k(t1,t2,k,time1.value) \
                #        * gref[br1,br2][time1,time2][:,0,:,0] # subtract loc part
                eps_gimp[br1,br2][time1,time2,k] = eps_s2p_K[br1][time1,k][:,:] \
                                                       * gref[br1,br2][time1,time2][:,0,:,0] # subtract loc part
                #eps_gimp[z1,z2,k] = eps_k(t1,t2,k,time1.value) \
                #        * gref[z1,z2][:,0][:,0] # subtract loc part
                #gimp_eps[br1,br2][time1,time2,k] = gref[br1,br2][time1,time2][:,0,:,0]  \
                #                                   * eps_k(t1,t2,k,time2.value)
                eps_gimp_eps[br1,br2][time1,time2,k] = eps_gimp[br1,br2][time1,time2,k] \
                                                       * eps_s2p_K[br2][time2,k][:,:] # subtract loc part (eps_loc) from eps_k
                gimp_eps[br1,br2][time1,time2,k] = gref[br1,br2][time1,time2][:,0,:,0]  \
                                                       * eps_s2p_K[br2][time2,k][:,:] # subtract loc part (eps_loc) from eps_k
"""
eps_gimp = eps_s2p_K @ gimp
eps_gimp_eps = eps_gimp @ eps_s2p_K
gimp_eps = gimp @ eps_s2p_K

eps_gimp_delta = eps_gimp @ delta
delta_gimp_eps = delta @ gimp_eps
delta_gimp = delta @ gimp
delta_gimp_delta = delta_gimp @ delta

#gimp_delta = gimp @ delta
#delta_gimp_delta = delta @ gimp_delta
#eps_gimp_delta = conv(eps_gimp,delta, [(1, 0)]) # [(0, 1), (1, 0)]
#delta_gimp_eps = conv(delta,gimp_eps, [(1, 0)])
#delta_gimp = conv(delta,gimp,[(1, 0)])#, [(0, 1), (1, 0)])
#delta_gimp_delta = conv(delta_gimp,delta,[(1, 0)])# [(0, 1), (1, 0)])
"""
Q = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches: 
    for br2 in branches:
        for k in bz_mesh:
            #print('k: ',k[0])
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                Q[br1,br2][time1,time2,k] = eps_gimp_eps[br1,br2][time1,time2,k] \
                                            - eps_gimp_delta[br1,br2][time1,time2,k] \
                                            - delta_gimp_eps[br1,br2][time1,time2,k] \
                                            + delta_gimp_delta[br1,br2][time1,time2] 
                #Q[br1,br2][time1,time2,k] = delta_gimp_delta[br1,br2][time1,time2] \
                #                            - eps_gimp_delta[br1,br2][time1,time2,k] \
                #                            - delta_gimp_eps[br1,br2][time1,time2,k]
                #Q[br1,br2][time1,time2,k] = eps_gimp_eps[br1,br2][time1,time2,k] \
                #                            - eps_gimp_delta[br1,br2][time1,time2,k] \
                #                            - delta_gimp_eps[br1,br2][time1,time2,k]
"""
Q = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches: 
    for br2 in branches:
        for k in bz_mesh:
            #print('k: ',k[0])
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                Q[br1,br2][time1,time2,k] = - delta[br1,br2][time1,time2] \
                                            + eps_gimp_eps[br1,br2][time1,time2,k] \
                                            - delta_gimp_eps[br1,br2][time1,time2,k]

# eps_tilde = eps_s2p_K - delta # TODO create new object (maybe)

#Q = eps_gimp_eps - eps_gimp_delta - delta_gimp_eps # + delta_gimp_delta  <- include

Q = 0.5 * (Q + herm_conj(Q)) # for now to circumvent the hermicity check !!!!

Q_herm = herm_conj(Q)

print(Q[FW,FW].data[0,0,:,:])
print(Q_herm[FW,FW].data[0,0,:,:])
print(Q[FW,FW].data[0,1,:,:])
print(Q_herm[FW,FW].data[0,1,:,:])
print(Q[FW,FW].data[1,0,:,:])
print(Q_herm[FW,FW].data[1,0,:,:])
print(Q[FW,FW].data[1,1,:,:])
print(Q_herm[FW,FW].data[1,1,:,:])

print('Q is Hermitian:', Q.is_hermitian())
print('gimp is Hermitian:',gimp.is_hermitian())
print('delta is Hermitian:',delta.is_hermitian())
print('delta_gimp: ', delta_gimp.is_hermitian())
print('gimp_eps is Hermitian:', gimp_eps.is_hermitian())
print('eps_gimp is Hermitian:', eps_gimp.is_hermitian())
print('eps_gimp_eps is Hermitian:', eps_gimp_eps.is_hermitian())
print('delta_gimp_delta is Hermitian:', delta_gimp_delta.is_hermitian())
print('eps_gimp_delta: ', eps_gimp_delta.is_hermitian())
print('delta_gimp_eps: ', delta_gimp_eps.is_hermitian())
print('- eps_gimp_delta - delta_gimp_eps: ',(- eps_gimp_delta - delta_gimp_eps).is_hermitian())
#print('delta_gimp_delta - eps_gimp_delta - delta_gimp_eps: ',is_hermitian(delta_gimp_delta - (eps_gimp_delta + delta_gimp_eps)))
print((delta_gimp_delta)[FW,FW].data[0,0,:,:])
print((delta_gimp_delta)[FW,FW].data[0,1,:,:])
print((delta_gimp_delta)[FW,FW].data[1,0,:,:])
print((delta_gimp_delta)[FW,FW].data[1,1,:,:])

print('t_max/n_t: ', t_max/n_t)
diff = delta_gimp_delta - herm_conj(delta_gimp_delta)
print(np.abs(diff[FW,FW].data[0,0,0,0]))
print(np.abs(diff[FW,FW].data[0,1,0,0]))
print(np.abs(diff[FW,FW].data[1,0,0,0]))
print(np.abs(diff[FW,FW].data[1,1,0,0]))

print(t_max/n_t, np.abs(diff[FW,FW].data[0,0,0,0]), np.abs(diff[FW,FW].data[0,1,0,0]), np.abs(diff[FW,FW].data[1,0,0,0]), np.abs(diff[FW,FW].data[1,1,0,0]))
#exit()
#print((eps_gimp_delta + delta_gimp_eps)[FW,FW].data[0,0,1,:,:])
#print((eps_gimp_delta + delta_gimp_eps)[FW,FW].data[0,0,2,:,:])
#print((eps_gimp_delta + delta_gimp_eps)[FW,FW].data[0,2,0,:,:])
#print((eps_gimp_delta + delta_gimp_eps)[FW,FW].data[0,2,1,:,:])
#print((eps_gimp_delta + delta_gimp_eps)[FW,FW].data[0,2,2,:,:])
#print(delta_gimp_delta[FW,FW].data.shape)

F = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                F[br1,br2][time1,time2,k] = - eps_gimp[br1,br2][time1,time2,k] \
                                            + delta_gimp[br1,br2][time1,time2]


print('F is Hermitian:', F.is_hermitian())
F = 0.5 * (F + herm_conj(F)) # for now to circumvent the check !!!!

Gd0_reg = solve_vie2(F, Q)

########################## CHECK GD0 ##############################
Gd0Gd0_h = herm_conj(Gd0_reg @ Gd0_reg)
Gd0_hGd0_h = herm_conj(Gd0_reg) @ herm_conj(Gd0_reg)

#Gd0_reg = solve_vie2(F, Q)
print(Gd0_reg[FW,BW].data.shape) #(21, 21, 9, 2, 2)
#exit()

safe = False
plot = False
if safe == True:

    with HDFArchive('Gd0Gd0_h_{}.h5'.format(n_t), 'w') as ar:
        ar['Gd0Gd0_h'] = Gd0Gd0_h[FW,BW].data[:,0,0,0,0].real
        ar['Gd0_hGd0_h'] = Gd0_hGd0_h[FW,BW].data[:,0,0,0,0].real

    with HDFArchive('Gd0Gd0_h_31.h5', 'r') as ar:
        Gd0Gd0_h_loaded_31 = ar['Gd0Gd0_h'][:]
        Gd0_hGd0_h_loaded_31 = ar['Gd0_hGd0_h'][:]

if plot == True:
    time_array_31 = np.linspace(0, t_max, 31)
    plt.plot(time_array_31, Gd0Gd0_h_loaded_31)
    plt.plot(time_array_31, Gd0_hGd0_h_loaded_31)

    with HDFArchive('Gd0Gd0_h_41.h5', 'r') as ar:
        Gd0Gd0_h_loaded_41 = ar['Gd0Gd0_h'][:]
        Gd0_hGd0_h_loaded_41 = ar['Gd0_hGd0_h'][:]

    time_array_41 = np.linspace(0, t_max, 41)
    plt.plot(time_array_41, Gd0Gd0_h_loaded_41)
    plt.plot(time_array_41, Gd0_hGd0_h_loaded_41)

    with HDFArchive('Gd0Gd0_h_51.h5', 'r') as ar:
        Gd0Gd0_h_loaded_51 = ar['Gd0Gd0_h'][:]
        Gd0_hGd0_h_loaded_51 = ar['Gd0_hGd0_h'][:]

    time_array_51 = np.linspace(0, t_max, 51)
    plt.plot(time_array_51, Gd0Gd0_h_loaded_51)
    plt.plot(time_array_51, Gd0_hGd0_h_loaded_51)

time_array = np.linspace(0, t_max, n_t)

plt.plot(time_array, Gd0Gd0_h[FW,BW].data[:,0,0,0,0].real)
plt.plot(time_array, Gd0_hGd0_h[FW,BW].data[:,0,0,0,0].real)

plt.savefig('Gd0_test.pdf')


print('t_max/n_t: ', t_max/n_t)
diffQQ_h = Q + herm_conj(Q)
print('diff QQ_h [0,0,0,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[0,0,0,0,0]))
print('diff QQ_h [1,0,0,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[1,0,0,0,0]))
print('diff QQ_h [0,1,0,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[0,1,0,0,0]))
print('diff QQ_h [1,1,0,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[1,1,0,0,0]))
print('#####################')
print('diff QQ_h [0,0,1,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[0,0,1,0,0]))
print('diff QQ_h [1,0,1,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[1,0,1,0,0]))
print('diff QQ_h [0,1,1,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[0,1,1,0,0]))
print('diff QQ_h [1,1,1,0,0] (FW,FW): ', np.abs(diffQQ_h[FW,FW].data[1,1,1,0,0]))
print('#####################')
print('diff QQ_h [0,0,0,0,0] (FW,BW): ', np.abs(diffQQ_h[FW,BW].data[0,0,0,0,0]))
print('diff QQ_h [1,0,0,0,0] (FW,BW): ', np.abs(diffQQ_h[FW,BW].data[1,0,0,0,0]))
print('diff QQ_h [0,1,0,0,0] (FW,BW): ', np.abs(diffQQ_h[FW,BW].data[0,1,0,0,0]))
print('diff QQ_h [1,1,0,0,0] (FW,BW): ', np.abs(diffQQ_h[FW,BW].data[1,1,0,0,0]))
print('#####################')
print('Q[FW,FW].data[0,0,0,0,0]', Q[FW,FW].data[0,0,0,0,0])
print('Q_h[FW,FW].data[0,0,0,0,0]', herm_conj(Q)[FW,FW].data[0,0,0,0,0])
print('Q[FW,BW].data[0,0,0,0,0]', Q[FW,BW].data[0,0,0,0,0])
print('Q_h[FW,BW].data[0,0,0,0,0]', herm_conj(Q)[FW,BW].data[0,0,0,0,0])
print('Q[FW,FW].data[1,0,0,0,0]', Q[FW,FW].data[1,0,0,0,0])
print('Q_h[FW,FW].data[1,0,0,0,0]', herm_conj(Q)[FW,FW].data[1,0,0,0,0])
print('Q[FW,FW].data[0,1,0,0,0]', Q[FW,FW].data[0,1,0,0,0])
print('Q_h[FW,FW].data[0,1,0,0,0]', herm_conj(Q)[FW,FW].data[0,1,0,0,0])
#diffGdGd_h = Gd0_reg-herm_conj(Gd0_reg)
#print('diff GdGd_h: ', np.abs(diffGdGd_h[FW,FW].data[0,0,0,0,0]))
#print(t_max/n_t, np.abs(diffGdGd_h[FW,FW].data[0,0,0,0,0]), np.abs(diffGdGd_h[FW,FW].data[0,1,0,0,0]), np.abs(diffGdGd_h[FW,FW].data[1,0,0,0,0]), np.abs(diffGdGd_h[FW,FW].data[1,1,0,0,0]))
#exit()
#t0 = list(t_mesh)[0]
#z = ContourPoint(Branch.FORWARD, t0)
#dt = t_max/n_t
## Open the file for writing
#with open('diffGG_h', 'a') as file:
#    # Write the two numbers to the file separated by a tab
#    file.write('{}\t{}\n'.format(dt,np.abs(I[FW,FW].data[0,0,0,0,0])))


print('Q is Hermitian:', Q.is_hermitian())
print('gimp is Hermitian:', gimp.is_hermitian())
print('delta is Hermitian:', delta.is_hermitian())
print('gimp_eps is Hermitian:', gimp_eps.is_hermitian())
print('eps_gimp is Hermitian:', eps_gimp.is_hermitian())
print('eps_gimp_eps is Hermitian:', eps_gimp_eps.is_hermitian())


########################## CHECK GD0 END ##############################
""" for testing susc_imp
# compute_keldysh_conn_correlator_2t(A, B, init_state, h, t_mesh, params)
# <T (A(t) - <A(t)>) (B(t') - <B(t')>)>

# Impurity susceptibility

# trying to use KeldyshGF.from_arg_index_gen() to build impurity susceptibility:

susc_imp_ch_test = -1j * compute_keldysh_conn_correlator_2t((n('up',0) + n('dn',0)),
                                        (n('up',0) + n('dn',0)),
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)


susc_imp_sp_test = -1j * compute_keldysh_conn_correlator_2t((n('up',0) - n('dn',0)),
                                        (n('up',0) - n('dn',0)),
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)



arg_index_shapes = ((2,), # ch = 0, sp = 1
                    (2,))

susc_imp_test = KeldyshGF(mesh=MeshProduct(t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)

for br1 in branches:
    for br2 in branches:
            susc_imp_test[br1,br2].data[:,:,0,0] = susc_imp_ch_test[br1,br2].data
            susc_imp_test[br1,br2].data[:,:,1,1] = susc_imp_sp_test[br1,br2].data

print(susc_imp_test[FW,FW].data[0,0,1,0])
"""

arg_index_shapes = ((2,), # ch = 0, sp = 1
                    (2,))

# Generator of scalar-valued elements
def generator_susc_imp(ind1, ind2):
    channel1 = ind1
    channel2 = ind2

    def get_operator(ind1, ind2):
        if ind1 == ind2 and ind1 == (0,):
            return (n('up',0) + n('dn',0))
        elif ind1 == ind2 and ind1 == (1,):
            return (n('up',0) - n('dn',0))
        else:
            return None

    operator = get_operator(ind1, ind2)
    #print(operator)
    if operator:
        #print('operator after if: ', operator)
        g_el = -1j * compute_keldysh_conn_correlator_2t(operator,
                                        operator,
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)
        return g_el
    else:
        g_el = KeldyshGF(mesh=tt_mesh)
        return g_el

susc_imp = KeldyshGF.from_arg_index_gen(generator_susc_imp, mesh=MeshProduct(t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)

susc_imp_U = KeldyshGF(mesh=MeshProduct(t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)
for br1 in branches:
    for br2 in branches:
            susc_imp_U[br1,br2].data[:,:,0,0] = susc_imp[br1,br2].data[:,:,0,0]*Uch
            susc_imp_U[br1,br2].data[:,:,1,1] = susc_imp[br1,br2].data[:,:,1,1]*Usp

# Impurity polarization
susc_imp = 0.5 * (susc_imp + herm_conj(susc_imp)) # for now to circumvent the hermicity check !!!!
susc_imp_U = 0.5 * (susc_imp_U + herm_conj(susc_imp_U))

pi_imp = solve_vie2(susc_imp_U, susc_imp)

# Vertex
arg_index_shapes = ((2,), #((2, 3),    # \sigma_1, l_1
                    (2,), #(2, 3),    # \sigma_2, l_2
                    (2,)) #(4, 3, 3)) # \varsigma, l_3, l_4

# Generator of scalar-valued elements
def generator_three_point_vertex(ind1, ind2, ind3):
    spin1 = ind1
    spin2 = ind2
    channel = ind3

    #def get_operator(ind1, ind2, ind3):
    c_index = ('up', 0) if spin1 == (0,) else ('dn', 0)
    c_dag_index = ('up', 0) if spin2 == (0,) else ('dn', 0)
    n_op = n('up', 0) + n('dn', 0) if channel == (0,) else n('up', 0) - n('dn', 0)

    #print('spin1, spin2, channel: ', spin1, spin2, channel)
    #print('c_index: ', c_index)
    #print('c_dag_index: ', c_dag_index)
    #print('n_op: ', n_op)

    g_el = compute_keldysh_vertex3(c_index, # c_indices
                                   c_dag_index , # c_dag_indices
                                   n_op, # n_op
                                   init_state,
                                   h,
                                   t_mesh,
                                   params)
    #g_el = KeldyshGF(mesh=ttt_mesh)
    return g_el


three_point_corr = KeldyshGF.from_arg_index_gen(generator_three_point_vertex, mesh=MeshProduct(t_mesh, t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)


U_pi_imp = KeldyshGF(mesh=MeshProduct(t_mesh, t_mesh), arg_index_shapes=((2,),(2,)))
for br1 in branches:
    for br2 in branches:
            U_pi_imp[br1,br2].data[:,:,0,0] = Uch*pi_imp[br1,br2].data[:,:,0,0]
            U_pi_imp[br1,br2].data[:,:,1,1] = Usp*pi_imp[br1,br2].data[:,:,1,1]

three_point_corr_U_pi_imp = conv(three_point_corr, U_pi_imp,
              [(2, 0)])

Lambda_test = three_point_corr @ three_point_corr_U_pi_imp #TODO: Check if Lambda same as Lambda_test

Lambda = three_point_corr - three_point_corr_U_pi_imp

#Uq_s2p_K[FW][t,k][0,0]
# Wprime
"""
Uq_piimp = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Uq_piimp_Uq = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))

for br1 in branches:
    for br2 in branches:
        for ch in range(2):
            for k in bz_mesh:
                for time1, time2 in MeshProduct(t_mesh, t_mesh):
                    #Uq_piimp[br1,br2][time1,time2,k][ch,ch] = Uq[k.index[0],k.index[1],k.index[2],ch] \
                    #                                     * pi_imp[br1,br2][time1,time2][ch,ch]
                    #Uq_piimp_Uq[br1,br2][time1,time2,k][ch,ch] = Uq_piimp[br1,br2][time1,time2,k][ch,ch] \
                    #                                      * Uq[k.index[0],k.index[1],k.index[2],ch]
                    Uq_piimp[br1,br2][time1,time2,k][ch,ch] = Uq_s2p_K[br1][time1,k][ch,ch] \
                                                         * pi_imp[br1,br2][time1,time2][ch,ch]
                    Uq_piimp_Uq[br1,br2][time1,time2,k][ch,ch] = Uq_piimp[br1,br2][time1,time2,k][ch,ch] \
                                                          * Uq_s2p_K[br2][time2,k][ch,ch]
"""

Uq_piimp = Uq_s2p_K @ pi_imp
Uq_piimp_Uq = Uq_piimp @ Uq_s2p_K

#print(Uq_piimp_Uq[FW,BW].data[0,3,4,1,1])

#Uq_piimp = 0.5 * (Uq_piimp + herm_conj(Uq_piimp)) # for now to circumvent the hermicity check !!!!
Uq_piimp_Uq = 0.5 * (Uq_piimp_Uq + herm_conj(Uq_piimp_Uq))
W0prime = solve_vie2(-Uq_piimp, Uq_piimp_Uq)

########################### diagrams #####################################

########################### diagrams  prepare #####################################

#self-energy
##g_fft = np.fft.ifftn(g_reshaped, axes=(0,1,2))
#eps_R = np.empty_like(eps_matrix)
#print(eps_R.shape)
#for time in t_mesh:
#    eps_K = eps_matrix[:,:,:,time.index]
#   #eps_fft = np.fft.ifftn(eps_K, axes=(0,1))
#    #eps_R[:,:,:,time.index] = eps_fft
#    eps_R[:,:,:,time.index] = np.fft.ifftn(eps_K, axes=(0,1,2))

#eps_R = Gf(mesh=tk_mesh, target_shape=(2, 2))
eps_s2p_R = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
eps_s2p_mR = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
eps_s2p_loc = Singular2PKeldyshGF(mesh=t_mesh, arg_index_shapes=((2,), (2,)))
print(eps_s2p_K)
print(eps_s2p_R)

for t, k in tk_mesh:
    print('eps_sp2 1: ', eps_s2p_K[FW][t,k])
    print('eps_sp2 2: ', eps_s2p_K[FW][t,:])
    print('eps_sp2 1a: ', eps_s2p_K[FW][t,k][0,0])
    print('eps_sp2 2a: ', eps_s2p_K[FW][t,:][0,0])
    print('eps_sp2 1b: ', eps_s2p_K[FW][t,k][:,:])
    print('eps_sp2 2b: ', eps_s2p_K[FW][t,:][:,:])
    print('eps_sp2 3: ', eps_s2p_K[FW][t,:].data)
    print('eps_sp2 data 3a: ', eps_s2p_K[FW][t,:][0,0].data)
    print('eps_sp2 data 3b: ', eps_s2p_K[FW][t,:][:,:].data)
    print('eps_sp2 data 4: ', eps_s2p_K[FW][t,k][0,0].data)
    print('eps_sp2 data 5: ', eps_s2p_K[FW][t,:][0,0].data.reshape(nkx*nky*nkz))
    print('eps_sp2_R data 6: ', eps_s2p_R[FW][t,:][0,0].data)
    #print('eps_s2p_K 6: ', eps_s2p_K[FW][t,:][0,0].reshape(4))
    #print('eps_s2p_R data: ', eps_s2p_R[FW][t,k][0,0].data.reshape(4))

#exit()
#print(eps_R)
#print(eps_s2p_K)
#print(dir(eps_R))
#print(dir(eps_s2p_K))
for time in t_mesh:
    for sigm in range(2):
        for br in branches:
            eps_s2p_loc[br][time][sigm,sigm] = np.mean(eps_s2p_K[br1][time,:][sigm,sigm].data)
            #eps_K = eps_matrix[:,:,:,time.index]
            #eps_fft = np.fft.ifftn(eps_K, axes=(0,1))
            #eps_R[:,:,:,time.index] = eps_fft
            #eps_K_FW = eps_s2p_K[FW][time,:][sigm1,sigm2].data.reshape(nkx,nky,nkz)
            #eps_K_BW = eps_s2p_K[BW][time,:][sigm1,sigm2].data.reshape(nkx,nky,nkz)
            eps_K = eps_s2p_K[br][time,:][sigm,sigm].data.reshape(nkx,nky,nkz)
            #print(eps_K_FW)
            
            #eps_R[t,:][sigm1,sigm2] = np.fft.ifftn(eps_s2p_K[FW][t,:][sigm1,sigm2].data.reshape(nkx,nky,nkz), axes=(0,1,2)).reshape(nkx*nky*nkz)
            #eps_s2p_R[FW].data[time.linear_index,:,sigm1,sigm2] = np.fft.ifftn(eps_K_FW, axes=(0,1,2)).reshape(nkx*nky*nkz)
            #eps_s2p_R[BW].data[time.linear_index,:,sigm1,sigm2] = np.fft.ifftn(eps_K_BW, axes=(0,1,2)).reshape(nkx*nky*nkz)
            #print(eps_s2p_R[FW][time,:][sigm1,sigm2].data)
            eps_s2p_R[br].data[time.linear_index,:,sigm,sigm] = np.fft.ifftn(eps_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k -> R 
            eps_s2p_mR[br].data[time.linear_index,:,sigm,sigm] = np.fft.fftn(eps_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k+q -> mR 
            #eps_s2p_R[FW][time,:][sigm1,sigm2] = np.fft.ifftn(eps_K_FW, axes=(0,1,2)).reshape(nkx*nky*nkz)
            #eps_s2p_R[BW][time,:][sigm1,sigm2] = np.fft.ifftn(eps_K_BW, axes=(0,1,2)).reshape(nkx*nky*nkz)
print(eps_s2p_R[FW].data[0,:,0,0].reshape(nkx,nky,nkz))

k_points = list(bz_mesh)
q0 = k_points[0]
print(q0)

Uq_tilde_s2p_R = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
Uq0_tilde = Singular2PKeldyshGF(mesh=t_mesh, arg_index_shapes=((2,), (2,)))
for time in t_mesh:
    for ch in range(2):
        for br in branches:
            Uq_tilde_K = Uq_tilde_s2p_K[br][time,:][ch,ch].data.reshape(nkx,nky,nkz)
            Uq0_tilde[br][time][ch,ch] = Uq_tilde_s2p_K[br][time,q0][ch,ch]
            #if ch == 0: # Uch
            #    Uq_tilde = Uq - 0.5*Uch
            #    Uq0_tilde[br][time][ch,ch] = Uq_s2p_K[br][time,q0][ch,ch] - 0.5*Uch
            #else: # Usp
            #    Uq_tilde = Uq - 0.5*Usp
            #    Uq0_tilde[br][time][ch,ch] = Uq_s2p_K[br][time,q0][ch,ch] - 0.5*Usp
            Uq_tilde_s2p_R[br].data[time.linear_index,:,ch,ch] = np.fft.ifftn(Uq_tilde_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R 


Gd0_reg_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Gd0_reg_mR = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Gd0_reg_loc = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                Gd0_reg_loc[br1,br2][time1,time2][sigm,sigm] = np.mean(Gd0_reg[br1,br2][time1,time2,:][sigm,sigm].data)
                Gd0_reg_K = Gd0_reg[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
                Gd0_reg_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(Gd0_reg_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R 
                Gd0_reg_mR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.fftn(Gd0_reg_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> -R 


# TODO this won't work if k-mesh does not start from Gamma point
# TODO check what to use for W and Uq np.fft.ifftn or np.fft.fftn (check also for Gd and eps) !!!!!!


print(Lambda[FW,FW,FW].data.shape)
print(dir(Gd0_reg_R))
#print(Gd0_reg_R.n_args)
print(Gd0_reg_R[FW,FW].data.shape)

########################### diagrams  prepare  END  #####################################

########################### Polarization #####################

Lambdaeps_s2p_mR = KeldyshGF(mesh=tttk_mesh, arg_index_shapes=((2,), (2,), (2,)))
eps_s2p_RLambda = KeldyshGF(mesh=tttk_mesh, arg_index_shapes=((2,), (2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for br3 in branches:
            for sigm in range(2): # spin up and dn
                for ch in range(2): # channel
                    for k in bz_mesh:
                        for time1, time2, time3 in MeshProduct(t_mesh, t_mesh, t_mesh):
                            Lambdaeps_s2p_mR[br1,br2,br3][time1,time2,time3,k][sigm, sigm, ch] = \
                                            Lambda[br1,br2,br3][time1,time2,time3][sigm, sigm, ch] \
                                            * eps_s2p_mR[br2][time2,k][sigm,sigm]
                            eps_s2p_RLambda[br1,br2,br3][time1,time2,time3,k][sigm, sigm, ch] = \
                                            Lambda[br1,br2,br3][time1,time2,time3][sigm, sigm, ch] \
                                            * eps_s2p_R[br2][time2,k][sigm,sigm]

# TODO: !!!!! check  eps_s2p mR vs R which one is which !!!!

Lambdaeps_s2p_R = conv(Lambda, eps_s2p_R,
             [(1, 0)])

eps_s2p_mRLambda = conv(eps_s2p_mR, Lambda,
             [(0, 1)])


LambdaGd0_reg_R = conv(Lambda, Gd0_reg_R,
             [(1, 0)])
Gd0_reg_mRLambda = conv(Gd0_reg_mR, Lambda,
             [(0, 1)])

Pi_R_1 = conv(LambdaGd0_reg_R, Gd0_reg_mRLambda,
             [(0, 0), (2, 1)])

Pi_R_2 = conv(Lambdaeps_s2p_R, Gd0_reg_mRLambda,
             [(0, 0),(2, 1)])

Pi_R_3 = conv(LambdaGd0_reg_R, eps_s2p_mRLambda,
             [(0, 0), (2, 1)])

Pi_R_4 = conv(Lambdaeps_s2p_R, eps_s2p_mRLambda,
             [(0, 0), (2, 1)])

Pi_R = Pi_R_1 + Pi_R_2 + Pi_R_3 + Pi_R_4

del LambdaGd0_reg_R, Lambdaeps_s2p_mR, Gd0_reg_mRLambda, eps_s2p_RLambda

Pi_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                Pi = Pi_R[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
                Pi_K[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(Pi, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R 

########################### Polarization END #####################

########################### Full W #####################
W0prime_Pi_K = W0prime @ Pi_K
Uq_tilde_s2p_K_Pi_K = Uq_tilde_s2p_K @ Pi_K

mFW = W0prime_Pi_K + Uq_tilde_s2p_K_Pi_K
mQW = mFW @ Uq_tilde_s2p_K
QW = W0prime - mQW
QW = 0.5 * (QW + herm_conj(QW)) # for now to circumvent the hermicity check !!!!
mFW = 0.5 * (mFW + herm_conj(mFW)) # for now to circumvent the hermicity check !!!!
Wprime = solve_vie2(-mFW, QW)

Wprime_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
W0prime_q0 = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for ch in range(2): # channel charge and spin
                W0prime_q0[br1,br2][time1,time2][ch,ch] = W0prime[br1,br2][time1,time2,q0][ch,ch]
                Wprime_K = Wprime[br1,br2][time1,time2,:][ch,ch].data.reshape(nkx,nky,nkz)
                Wprime_R[br1,br2].data[time1.linear_index,time2.linear_index,:,ch,ch] = np.fft.ifftn(Wprime_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k+q -> R 

########################### Full W END #####################
########################### Self-Energy #####################################
"""
Lambdaeps_s2p_R = KeldyshGF(mesh=tttk_mesh, arg_index_shapes=((2,), (2,), (2,)))
Uq_tilde_s2p_RLambda = KeldyshGF(mesh=tttk_mesh, arg_index_shapes=((2,), (2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for br3 in branches:
            for sigm in range(2): # spin up and dn
                for ch in range(2): # channel
                    for k in bz_mesh:
                        for time1, time2, time3 in MeshProduct(t_mesh, t_mesh, t_mesh):
                            Lambdaeps_s2p_R[br1,br2,br3][time1,time2,time3,k][sigm, sigm, ch] = \
                                            Lambda[br1,br2,br3][time1,time2,time3][sigm, sigm, ch] \
                                            * eps_s2p_R[br2][time2,k][sigm,sigm]
                            Uq_tilde_s2p_RLambda[br1,br2,br3][time1,time2,time3,k][sigm, sigm, ch] = \
                                            Uq_tilde_s2p_R[br3][time3,k][ch,ch] \
                                            * Lambda[br1,br2,br3][time1,time2,time3][sigm, sigm, ch]
"""
LambdaGd0_reg_mR = conv(Lambda, Gd0_reg_mR,
             [(1, 0)])

print(LambdaGd0_reg_mR.n_args)
print(LambdaGd0_reg_mR[FW,FW,FW].data.shape)

Wprime_RLambda = conv(Wprime_R, Lambda,
             [(1, 2)])

print(Wprime_RLambda.n_args)
print(Wprime_RLambda[FW,FW,FW].data.shape)

Lambdaeps_s2p_mR = conv(Lambda, eps_s2p_mR,
             [(1, 0)])
Uq_tilde_s2p_RLambda = conv(Uq_tilde_s2p_R, Lambda,
             [(1, 2)])

sigma_R_1 = conv(LambdaGd0_reg_mR, Wprime_RLambda,
             [(1, 0), (2, 1)])                            

sigma_R_2 = conv(Lambdaeps_s2p_mR, Wprime_RLambda,
               [(1, 0), (2, 1)])

sigma_R_3 = conv(LambdaGd0_reg_mR, Uq_tilde_s2p_RLambda,
               [(1, 0), (2, 1)])

sigma_R_4 = conv(Lambdaeps_s2p_mR, Uq_tilde_s2p_RLambda,
               [(1, 0), (2, 1)])

sigma_R = 1j * (sigma_R_1 + sigma_R_2 + sigma_R_3 + sigma_R_4)

del LambdaGd0_reg_mR, Wprime_RLambda, Lambdaeps_s2p_mR, Uq_tilde_s2p_RLambda

sigma_dual_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                sigma = sigma_R[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
                sigma_dual_K[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(sigma, axes=(0,1,2)).reshape(nkx*nky*nkz) # R -> K 

############# Tadpole #####################
"""
#Lambdaeps_s2p_loc = KeldyshGF(mesh=ttt_mesh, arg_index_shapes=((2,), (2,), (2,)))
Lambdaeps_s2p_loc = KeldyshGF(mesh=t_mesh, arg_index_shapes=((2,),))
LambdaUq0_tilde = KeldyshGF(mesh=ttt_mesh, arg_index_shapes=((2,), (2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for br3 in branches:
            for sigm in range(2): # spin up and dn
                for ch in range(2): # channel
                    for time1, time2, time3 in MeshProduct(t_mesh, t_mesh, t_mesh):
                        #Lambdaeps_s2p_loc[br1,br1,br3][time1,time1,time3][sigm, sigm, ch] = \
                        #                Lambda[br1,br1,br3][time1,time1,time3][sigm, sigm, ch] \
                        #                * eps_s2p_loc[br1][time1][sigm,sigm]
                        Lambdaeps_s2p_loc[br3][time3][ch] += \
                                        Lambda[br1,br1,br3][time1,time1,time3][sigm, sigm, ch] \
                                        * eps_s2p_loc[br1][time1][sigm,sigm]
                        LambdaUq0_tilde[br1,br2,br3][time1,time2,time3][sigm, sigm, ch] = \
                                        Lambda[br1,br2,br3][time1,time2,time3][sigm, sigm, ch] \
                                        * Uq0_tilde[br3][time3][ch,ch]
"""
#Lambdaeps_s2p_loc = conv(Lambda, eps_s2p_loc,
#             [(0, 1),(1, 0)])        

LambdaUq0_tilde = conv(Lambda, Uq0_tilde,
             [(2, 0)]) 

LambdaW0prime_q0 = conv(Lambda, W0prime_q0,
             [(2, 0)])   

LambdaGd0_reg_loc = conv(Lambda, Gd0_reg_loc,
             [(0, 1),(1, 0)])
  

sigma_tadpole_1 = conv(LambdaW0prime_q0, LambdaGd0_reg_loc,
             [(2, 0)]) 

#sigma_tadpole_2 = conv(LambdaWprime_q0, Lambdaeps_s2p_loc,
#             [(2, 0)]) 

sigma_tadpole_3 = conv(LambdaUq0_tilde, LambdaGd0_reg_loc,
             [(2, 0)]) 

#sigma_tadpole_4 = conv(LambdaUq0_tilde, Lambdaeps_s2p_loc,
#             [(2, 0)]) 

sigma_tadpole = -1j*(sigma_tadpole_1 + sigma_tadpole_3)
#sigma_tadpole = -1j*(sigma_tadpole_1 + sigma_tadpole_2 + sigma_tadpole_3 + sigma_tadpole_4)

del LambdaW0prime_q0, LambdaUq0_tilde, LambdaGd0_reg_loc #, Lambdaeps_s2p_loc

############# Tadpole END #####################

############# Full Self-Energy #####################

print(sigma_dual_K.mesh)
print(sigma_tadpole.mesh)

sigma_dual_full = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for sigm in range(2): # spin up and dn
            for time1, time2 in tt_mesh:
                for k in bz_mesh:
                    sigma_dual_full[br1,br2][time1,time2,k][sigm,sigm] = sigma_dual_K[br1,br2][time1,time2,k][sigm,sigm] + sigma_tadpole[br1,br2][time1,time2][sigm,sigm]

############# Full Self-Energy END #####################

########################### Self-Energy END #####################################

K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                K[br1,br2][time1,time2,k] = sigma_dual_full[br1,br2][time1,time2,k][:,:] \
                                                       + gref[br1,br2][time1,time2][:,0,:,0]
"""
Keps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                #eps_gimp[br1,br2][time1,time2,k] = eps_k(t1,t2,k,time1.value) \
                #        * gref[br1,br2][time1,time2][:,0,:,0] # subtract loc part
                K[br1,br2][time1,time2,k] = K[br1,br2][time1,time2,k]  \
                                            * eps_s2p_K[br2][time2,k]
"""

Keps= K @ eps_s2p_K
Kdelta = K @ delta      

FG = Kdelta - Keps
FG = 0.5 * (FG + herm_conj(FG)) # for now to circumvent the hermicity check !!!!
K = 0.5 * (K + herm_conj(K)) # for now to circumvent the hermicity check !!!!
G_latt = solve_vie2(FG, K)

print(G_latt[FW,FW].data[0,:,0,0,0])

"""
A = KeldyshGF(mesh=ttt_mesh, arg_index_shapes=((2, 3), (2, 4), (2, 5)))
B = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2, 4), (2, 6)))

C = conv(A, B,
         [(1, 0)])

print(C.n_args) # output 3
print(C.arg_index_shapes) # output ((2, 3), (2, 5), (2, 6))
"""
"""
A = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2, 3), (2, 4)))
B = KeldyshGF(mesh=ttt_mesh, arg_index_shapes=((2, 5), (2, 6), (2, 4)))

C = conv(A, B,
         [(1, 2)])

print(C.n_args) # output 3
print(C.arg_index_shapes) # output ((2, 3), (2, 5), (2, 6))
"""


exit()
Lambdaeps_R = Lambda
for br1 in branches:
    for br2 in branches:
        for br3 in branches:
            for sigm in range(2): # spin up and dn
                for sigm in range(2): # spin up and dn
                    for ch in range(2):
                        for k in bz_mesh:
                            for time1, time2, time3 in MeshProduct(t_mesh, t_mesh, t_mesh):
                                Lambdaeps_R[br1,br2,br3][time1,time2,time3][sigm,sigm,ch] *= eps_R[k.index[0], k.index[1], k.index[2],time2.linear_index]


LambdaWprime = conv(Lambda, Wprime, [(2, 1)], free_args=([0, 1], [2]))

diag3 = 1j * conv(Lambdaeps_R, LambdaWprime, [(1, 0), (2, 2)])

selfenergy_2nd_order(Lambda,Gd0_reg_test,Wprime)
exit()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# Points on the real time axis 
#t_points = list(t_mesh)
time1, time2 = t_points[0], t_points[1]
# Contour points
z1 = ContourPoint(Branch.FORWARD, time1)
z2 = ContourPoint(Branch.BACKWARD, time2)

print(gf)
print(dir(gf['dn']))
print(*gf)
print(gf['dn'].components.shape)
print(gf['dn'].components)
print(gf['dn'][FW,FW])
print(gf['dn'][FW,FW].data)
print('##########')
print(gf['dn'][z1,z2])
print(gf['dn'][z1,z2].data)
print('FW: ', FW)
print('BW: ', BW)
print('time1: ', time1)
print('time2: ', time2)
print('z1: ', z1)
print('z2: ', z2)

#print(gf[z1,z2])
#print(gf['dn'].shape)
#g=gf['dn'].components.reshape(4)


print(gf['up'].n_args)

#exit()
#print(type(gf))
#print(type(gf['dn']))
#print(type(tri_vertex_ch))

print(dir(gf['dn']))
print(gf['dn'].arg_index_shapes)
#print(g.arg_index_shapes)
#print(tri_vertex_ch.shape())
#print(tri_vertex_ch.arg_index_shapes)

#C = conv(gf['dn'], eps_k,
#         [(0, 1), (1, 0)])
#C = conv(gf['dn'], gf['dn'],
#         [(0, 1)])
print(gf['dn'].components.data.shape)

"""

# Vertex
arg_index_shapes = ((2,), #((2, 3),    # \sigma_1, l_1
                    (2,), #(2, 3),    # \sigma_2, l_2
                    (2,)) #(4, 3, 3)) # \varsigma, l_3, l_4

Lambda = KeldyshGF(mesh=MeshProduct(t_mesh, t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)


#density_f_0 = compute_expectval(n('up',0) + n('dn',0), init_state, h, t_mesh, params)

#print('density: ' ,density_f_0)
#print('n: ',n('up', 0) + n('dn', 0))
#rho = (n('up', 0) + n('dn', 0)) - density_f_0


#exit()
tri_vertex_ch_uu = compute_keldysh_vertex3(('up', 0), # c_indices
                                        ('up', 0), # c_dag_indices
                                        n('dn', 0) + n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

tri_vertex_ch_ud = compute_keldysh_vertex3(('up', 0), # c_indices
                                        ('dn', 0), # c_dag_indices
                                        n('dn', 0) + n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

tri_vertex_ch_du = compute_keldysh_vertex3(('dn', 0), # c_indices
                                        ('up', 0), # c_dag_indices
                                        n('dn', 0) + n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

tri_vertex_ch_dd = compute_keldysh_vertex3(('dn', 0), # c_indices
                                        ('dn', 0), # c_dag_indices
                                        n('dn', 0) + n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)


tri_vertex_sp_uu = compute_keldysh_vertex3(('up', 0), # c_indices
                                        ('up', 0), # c_dag_indices
                                        n('dn', 0) - n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

tri_vertex_sp_ud = compute_keldysh_vertex3(('up', 0), # c_indices
                                        ('dn', 0), # c_dag_indices
                                        n('dn', 0) - n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

tri_vertex_sp_du = compute_keldysh_vertex3(('dn', 0), # c_indices
                                        ('up', 0), # c_dag_indices
                                        n('dn', 0) - n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

tri_vertex_sp_dd = compute_keldysh_vertex3(('dn', 0), # c_indices
                                        ('dn', 0), # c_dag_indices
                                        n('dn', 0) - n('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

#tri_vertex_test = compute_keldysh_vertex3(('dn', 0), # c_indices
#                                        ('dn', 0), # c_dag_indices
#                                        c_dag('dn', 0)*c_dag('up', 0), # n_op
#                                        init_state,
#                                        h,
#                                        t_mesh,
#                                        params)

#tri_vertex_test2 = compute_keldysh_vertex3(('dn', 0), # c_indices
#                                        ('dn', 0), # c_dag_indices
#                                        rho, # n_op
#                                        init_state,
#                                        h,
#                                        t_mesh,
#                                        params)

print(Lambda[FW,BW,BW].data.shape) #(3, 3, 3, 2, 2, 4)
#print(Lambda[FW,BW,BW].data[0,0,0,0,0,0,0,0,0,0])
print('Lambda: ', Lambda)
print(tri_vertex_ch_uu)
print(tri_vertex_ch_uu[FW,BW,BW].data.shape)
print('tri_vertex_ch_uu: ', tri_vertex_ch_uu)


for br1 in branches:
    for br2 in branches:
        for br3 in branches:
            Lambda[br1,br2,br3].data[:,:,:,0,0,0] = tri_vertex_ch_uu[br1,br2,br3].data
            Lambda[br1,br2,br3].data[:,:,:,0,1,0] = tri_vertex_ch_ud[br1,br2,br3].data
            Lambda[br1,br2,br3].data[:,:,:,1,0,0] = tri_vertex_ch_du[br1,br2,br3].data
            Lambda[br1,br2,br3].data[:,:,:,1,1,0] = tri_vertex_ch_dd[br1,br2,br3].data

            Lambda[br1,br2,br3].data[:,:,:,0,0,1] = tri_vertex_sp_uu[br1,br2,br3].data
            Lambda[br1,br2,br3].data[:,:,:,0,1,1] = tri_vertex_sp_ud[br1,br2,br3].data
            Lambda[br1,br2,br3].data[:,:,:,1,0,1] = tri_vertex_sp_du[br1,br2,br3].data
            Lambda[br1,br2,br3].data[:,:,:,1,1,1] = tri_vertex_sp_dd[br1,br2,br3].data



print(Lambda[FW,FW,FW].data[0,0,0,0,0,0])
print(Lambda[FW,FW,FW].data[0,1,1,0,0,0])
print(Lambda[FW,FW,FW].data[1,0,1,0,0,0])
print(Lambda[FW,FW,FW].data[1,1,0,0,0,0])
print(Lambda[FW,FW,FW].data[1,1,1,0,0,0])

print(Lambda[FW,FW,FW].data[0,0,0,0,0,1])
print(Lambda[FW,FW,FW].data[0,1,1,0,0,1])
print(Lambda[FW,FW,FW].data[1,0,1,0,0,1])
print(Lambda[FW,FW,FW].data[1,1,0,0,0,1])
print(Lambda[FW,FW,FW].data[1,1,1,0,0,1])

print(Lambda[FW,FW,BW].data[0,0,0,0,0,0])
print(Lambda[FW,FW,BW].data[0,1,1,0,0,0])
print(Lambda[FW,FW,BW].data[1,0,1,0,0,0])
print(Lambda[FW,FW,BW].data[1,1,0,0,0,0])
print(Lambda[FW,FW,BW].data[1,1,1,0,0,0])
