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

from tddt.keldysh import Branch, KeldyshGF, from_lesser_greater, conv
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

from tddt.keldysh import Branch, lesser, greater, retarded, advanced
from tddt.keldysh import ContourPoint
from tddt.keldysh import herm_conj, is_hermitian

# Import some TRIQS modules related to lattice
from triqs.lattice import BravaisLattice, BrillouinZone
from triqs.gf import MeshBrillouinZone

from tddt.vie2 import solve_vie2

########################## Reference system #######################

# Model parameters
U = 3.0
Uch = U/2
Usp = -U/2
mu = 0.5 * U
#eps = 0.2
t1 = 1.0 # nearest neighbor hopping 
t2 = 0.0 # next nearest neighbor hopping
A = 0.1


# time-mesh
t_max = 10.0 #20.0
n_t = 6 #21
t_mesh = MeshReTime(0, t_max, n_t)
tt_mesh = MeshProduct(t_mesh, t_mesh) # A 2D mesh as a direct product of t_mesh with itself
m_interp = MeshReTime(0, t_max, n_t)


#k-mesh
lat = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # 2D square lattice
bz = BrillouinZone(lat)  # Brillouin zone of the lattice
n_k = 3 # Number of k-points along each dimension
bz_mesh = MeshBrillouinZone(bz, n_k) # k-mesh on 1BZ; 0 - 2pi

#tk_mesh = MeshProduct(t_mesh, bz_mesh)
ttk_mesh = MeshProduct(t_mesh, t_mesh, bz_mesh)


# Keldysh branches 
FW = Branch.FORWARD
BW = Branch.BACKWARD
branches = (FW, BW)


# time dependent hopping for Hamiltonian
dt_pos = ti(m_interp, np.array([t1*np.exp(1.j * A * np.cos(1 * x)) for x in m_interp]))
dt_neg = ti(m_interp, np.array([t1*np.exp(-1.j * A * np.cos(1 * x)) for x in m_interp]))
#print(dt_pos)
#print(*dt_pos.data)


# time dependent hopping for Hybridisation
# Let it as a function or better create an object (array) and use this throughout the calculation?
def V(axis,sign,t):
    """ axis = x,y
        sign = -1,+1
    """
    V = t1*np.exp(sign*1.j*A*np.cos(1 * t))
    return V


# Lattice despersion
def eps_k(t1,t2,k,time):
    eps_k = -2.0*t1*(np.cos(k[0]-A*np.cos(time))+np.cos(k[1]-A*np.cos(time))) \
            -4.0*t2*np.cos(k[0]-A*np.cos(time))*np.cos(k[1]-A*np.cos(time))
    return eps_k

eps_loc = np.zeros(n_t)

for time in t_mesh:
    for k in bz_mesh:
        eps_loc[time.index] += eps_k(t1,t2,k,time.value)
        #print(time.value)
        #print(np.cos(time.value))

eps_loc = eps_loc/(n_k*n_k)

print('eps_loc: ', eps_loc)

# Make Keldysh GF of a single fermion with energy `eps` and occupation number `occup_n`
def make_g(eps, occup_n, t_mesh):
    tt_mesh = MeshProduct(t_mesh, t_mesh)

    g_l = Gf(mesh=tt_mesh, target_shape=(1, 1))
    g_g = Gf(mesh=tt_mesh, target_shape=(1, 1))

    for time1, time2 in tt_mesh:
        g_g[time1, time2] = -1j * (1.0 - occup_n) * np.exp(-1j * eps * (time1 - time2))
        g_l[time1, time2] = -1j * (-occup_n) * np.exp(-1j * eps * (time1 - time2))

    return from_lesser_greater(g_l, g_g)


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

#h0 = -mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0)
#h0 += sum(V('y', 1) * c_dag(sn, 0) * c(sn, 1) + V('y', -1) * c_dag(sn, 1) * c(sn, 0)
#          for sn in spin_names)
#h0 += sum(V('x', 1) * c_dag(sn, 0) * c(sn, 2) + V('x', -1) * c_dag(sn, 2) * c(sn, 0)
#          for sn in spin_names)
#h0 += sum(V('y', -1) * c_dag(sn, 0) * c(sn, 3) + V('y', 1) * c_dag(sn, 3) * c(sn, 0)
#          for sn in spin_names)
#h0 += sum(V('x', -1) * c_dag(sn, 0) * c(sn, 4) + V('x', 1) * c_dag(sn, 4) * c(sn, 0)
#          for sn in spin_names)


init_state = make_equilibrium_init_state(h0,
                                         fermion_indices=fops,
                                         boson_indices=set(),
                                         temperature=0,
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
gf_struct = [('up', 1), ('dn', 1)]
gf = compute_keldysh_gf(gf_struct,
                        init_state,
                        h,
                        t_mesh,
                        params)

gimp = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
#print(gimp[FW,FW][0,0].data.shape)
#print(gf['up'][FW,FW].data.shape)
#print(gf)
#print(gf['up'])
#print(gf['up'][FW,FW])
#print(dir(gf))
#print(dir(gf['up']))
#print(dir(gf['up'][FW,FW]))

for br1 in branches:
    for br2 in branches:
        gimp[br1,br2][0,0].data[...] = -1j * gf['up'][br1,br2].data[...,0,0] # Is there a better way to deal with that (gf['up/dn'])?
        gimp[br1,br2][1,1].data[...] = -1j * gf['dn'][br1,br2].data[...,0,0] #TODO: multiply by -i (see eq. 32) ?????????


# GF of a noncorrelated site
eps = -mu
occup_n = 0.5
gf_non_corr_site = make_g(eps, occup_n, t_mesh)

#print('gimp: ', gimp.components)
#print('gf: ', gf['up'][FW,BW].data.shape)
#print('dt: ', dt.data.shape)


# Hybridisation function delta
delta = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))

for sp in (0,1):
    for br1 in branches:
        for br2 in branches:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                #print(ContourPoint(Branch.FORWARD, time1).t)
                #print(ContourPoint(Branch.FORWARD, time1).t.value)
                z1 = ContourPoint(br1, time1)
                z2 = ContourPoint(br2, time2)
                #delta[z1,z2] = dt.data[time1.linear_index]*gf_non_corr_site[z1,z2]*dt.data[time2.linear_index] # Sum over i=(1,2,3,4) is missing!!!

                delta[z1,z2][sp,sp] = V('x',+1,time1.linear_index) * gf_non_corr_site[z1,z2] * V('x',-1,time2.linear_index) \
                                      + V('x',-1,time1.linear_index) * gf_non_corr_site[z1,z2] * V('x',+1,time2.linear_index) \
                                      + V('y',-1,time1.linear_index) * gf_non_corr_site[z1,z2] * V('y',+1,time2.linear_index) \
                                      + V('y',+1,time1.linear_index) * gf_non_corr_site[z1,z2] * V('y',-1,time2.linear_index)



gimp_eps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_gimp = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_gimp_eps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                #print('time1: ', time1)
                #print('time2: ', time2)
                eps_gimp[br1,br2][time1,time2,k] = eps_k(t1,t2,k,time1.value) \
                                                   * gimp[br1,br2][time1,time2] # subtract loc part
                eps_gimp_eps[br1,br2][time1,time2,k] = eps_gimp[br1,br2][time1,time2,k] \
                                                       * eps_k(t1,t2,k,time2.value) # subtract loc part (eps_loc) from eps_k

eps_gimp_delta = eps_gimp @ delta
delta_gimp_eps = delta @ gimp_eps
delta_gimp = delta @ gimp
delta_gimp_delta = delta_gimp @ delta

Q = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            #print('k: ',k[0])
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                Q[br1,br2][time1,time2,k] = eps_gimp_eps[br1,br2][time1,time2,k] \
                                            + delta_gimp_delta[br1,br2][time1,time2] \
                                            - eps_gimp_delta[br1,br2][time1,time2,k] \
                                            - delta_gimp_eps[br1,br2][time1,time2,k]

Q = 0.5 * (Q + herm_conj(Q))

F = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                F[br1,br2][time1,time2,k] = eps_gimp[br1,br2][time1,time2,k] \
                                            - delta_gimp[br1,br2][time1,time2]

F_test = 0.5 * (F + herm_conj(F))

Gd0_test = solve_vie2(F_test, Q)

#Gd0 = solve_vie2(F, Q)

#exit()

nch = (n('up',0) + n('dn',0))
nsp = (n('up',0) - n('dn',0))

# compute_keldysh_conn_correlator_2t(A, B, init_state, h, t_mesh, params)
# <T (A(t) - <A(t)>) (B(t') - <B(t')>)>
# Impurity susceptibility

susc_imp_ch = -1j * compute_keldysh_conn_correlator_2t(nch, 
                                        nch,
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

susc_imp_sp = -1j * compute_keldysh_conn_correlator_2t(nsp,
                                        nsp,
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)


print('susc_imp_ch is Hermitian:',is_hermitian(susc_imp_ch))

#pi_imp_ch = solve_vie2(susc_imp_ch*Uch, susc_imp_ch)
pi_imp_sp = solve_vie2(susc_imp_sp*Uch, susc_imp_sp)

#exit()




diffQQ_h = Q-herm_conj(Q)
print('diff QQ_h: ', np.abs(diffQQ_h[FW,FW].data[0,0,0,0,0]))

#t0 = list(t_mesh)[0]
#z = ContourPoint(Branch.FORWARD, t0)
#dt = t_max/n_t
## Open the file for writing
#with open('diffGG_h', 'a') as file:
#    # Write the two numbers to the file separated by a tab
#    file.write('{}\t{}\n'.format(dt,np.abs(I[FW,FW].data[0,0,0,0,0])))


print('Q is Hermitian:',is_hermitian(Q))
print('gimp is Hermitian:',is_hermitian(gimp))
print('gimp_eps is Hermitian:',is_hermitian(gimp_eps))
print('eps_gimp_eps is Hermitian:',is_hermitian(eps_gimp_eps))
print('delta is Hermitian:',is_hermitian(delta))

#exit()

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
