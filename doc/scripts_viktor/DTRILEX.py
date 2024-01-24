import unittest
import pytest
from itertools import product
import numpy as np
from numpy.testing import assert_array_almost_equal

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
mu = 0.5 * U
#eps = 0.2
t1 = 1.0 # nearest neighbor hopping 
t2 = 0.0 # next nearest neighbor hopping
A = 0.1


# time-mesh
t_max = 5.0
n_t = 10
t_mesh = MeshReTime(0, t_max, n_t)
tt_mesh = MeshProduct(t_mesh, t_mesh) # A 2D mesh as a direct product of t_mesh with itself
m_interp = MeshReTime(0, t_max, 15)


#k-mesh
lat = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # 2D square lattice
bz = BrillouinZone(lat)  # Brillouin zone of the lattice
n_k = 3 # Number of k-points along each dimension
bz_mesh = MeshBrillouinZone(bz, n_k) # k-mesh on 1BZ; 0 - 2pi


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
    eps_k = -2.0*t1*(np.cos(k[0]-A*np.cos(time))+np.cos(k[1]-A*np.cos(time)))\
            -4.0*t2*np.cos(k[0]-A*np.cos(time))*np.cos(k[1]-A*np.cos(time))
    return eps_k


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

gf_imp = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
#print(gf_imp[FW,FW][0,0].data.shape)
#print(gf['up'][FW,FW].data.shape)
#print(gf)
#print(gf['up'])
#print(gf['up'][FW,FW])
#print(dir(gf))
#print(dir(gf['up']))
#print(dir(gf['up'][FW,FW]))

for br1 in branches:
    for br2 in branches:
        gf_imp[br1,br2][0,0].data[...] = gf['up'][br1,br2].data[...,0,0] # Is there a better way to deal with that (gf['up/dn'])?
        gf_imp[br1,br2][1,1].data[...] = gf['dn'][br1,br2].data[...,0,0]


# GF of a noncorrelated site
eps = -mu
occup_n = 0.5
gf_non_corr_site = make_g(eps, occup_n, t_mesh)

#print('gf_imp: ', gf_imp.components)
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

                delta[z1,z2][sp,sp] = V('x',+1,time1.linear_index)*gf_non_corr_site[z1,z2]*V('x',-1,time2.linear_index) \
                                    + V('x',-1,time1.linear_index)*gf_non_corr_site[z1,z2]*V('x',+1,time2.linear_index) \
                                    + V('y',-1,time1.linear_index)*gf_non_corr_site[z1,z2]*V('y',+1,time2.linear_index) \
                                    + V('y',+1,time1.linear_index)*gf_non_corr_site[z1,z2]*V('y',-1,time2.linear_index)

#exit()
####
eps_tild_k = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
gf_imp_on_k_mesh = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_tild_gf_imp_vareps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_tild_gf_imp_vareps_test = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))


#eps_k_test = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
#print(eps_k_test[FW,FW].data.shape)
#print('is Hermitian:',is_hermitian(eps_k_test))
#for br1 in branches:
#    for br2 in branches:
#        for k in (bz_mesh):
#            #print('k: ',k[0])
#            for time1, time2 in MeshProduct(t_mesh, t_mesh):
#                if time1.value == time2.value:
#                    #eps_k_test[br1,br2][time1, time2, k] = eps_k(t1,t2,k)
#                    #print(time1)
#                    #eps_tild_k[br1,br2][time1,time2,k] = eps_k(t1,t2,k,time1.value) - delta[br1,br2][time1,time2]
#                    eps_tild_k[br1,br2][time1,time2,k] = - delta[br1,br2][time1,time2]
#                else:
#                    eps_tild_k[br1,br2][time1,time2,k] = - delta[br1,br2][time1,time2]


g_impeps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
epsg_imp = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
epsg_impeps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in (bz_mesh):
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                #print('time1: ', time1)
                #print('time2: ', time2)
                g_impeps[br1,br2][time1,time2,k] = gf_imp[br1,br2][time1,time2] * eps_k(t1,t2,k,time2.value)
                epsg_imp[br1,br2][time1,time2,k] = eps_k(t1,t2,k,time1.value) * gf_imp[br1,br2][time1,time2]
                #print(g_impeps[br1,br2][time1,time2,k][:,:])
                epsg_impeps[br1,br2][time1,time2,k] = eps_k(t1,t2,k,time1.value) * g_impeps[br1,br2][time1,time2,k]
#gf_imp[br1,br2][time1,time2] * eps_k(t1,t2,k,time2.value)
                # no spin index because both diagonal in spin

print(g_impeps[FW,FW].data[0,0,0,:,:])
print(g_impeps[FW,FW].data[0,1,0,:,:])
print(g_impeps[FW,FW].data[1,0,0,:,:])
print(g_impeps[FW,FW].data[1,1,0,:,:])

C = -0.5 * delta @ g_impeps
Cb = -0.5 * g_impeps @ delta
C1 = delta @ gf_imp
D = C - Cb
E = D + epsg_impeps
Q = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
print('D is Hermitian:',is_hermitian(D))
print('E is Hermitian:',is_hermitian(E))
print('C1 is Hermitian:',is_hermitian(C1))
print('gf_imp is Hermitian:',is_hermitian(gf_imp))
print('g_impeps is Hermitian:',is_hermitian(g_impeps))
print('epsg_impeps is Hermitian:',is_hermitian(epsg_impeps))
print('C is Hermitian:',is_hermitian(C))
print('delta is Hermitian:',is_hermitian(delta))


#print('eps_tild_k is Hermitian:',is_hermitian(eps_tild_k))
#print('delta is Hermitian:',is_hermitian(delta))

#print('delta[FW,FW]:',g_impeps[FW,FW].data[0,0,:,:,:])
#print('delta[FW,BW]:',g_impeps[FW,BW].data[0,0,:,:,:])
#print('delta[BW,FW]:',g_impeps[BW,FW].data[0,0,:,:,:])
#print('delta[BW,BW]:',g_impeps[BW,BW].data[0,0,:,:,:])

#print('delta[FW,FW]:',g_impeps[FW,FW].data[0,1,:,:,:])
#print('delta[FW,BW]:',g_impeps[FW,BW].data[0,1,:,:,:])
#print('delta[BW,FW]:',g_impeps[BW,FW].data[0,1,:,:,:])
#print('delta[BW,BW]:',g_impeps[BW,BW].data[0,1,:,:,:])

#print('delta[FW,FW]:',g_impeps[FW,FW].data[1,0,:,:,:])
#print('delta[FW,BW]:',g_impeps[FW,BW].data[1,0,:,:,:])
#print('delta[BW,FW]:',g_impeps[BW,FW].data[1,0,:,:,:])

#print('delta[BW,BW]:',g_impeps[BW,BW].data[1,0,:,:,:])

#print('delta[FW,FW]:',g_impeps[FW,FW].data[1,1,:,:,:])
#print('delta[FW,BW]:',g_impeps[FW,BW].data[1,1,:,:,:])
#print('delta[BW,FW]:',g_impeps[BW,FW].data[1,1,:,:,:])
#print('delta[BW,BW]:',g_impeps[BW,BW].data[1,1,:,:,:])


for br1 in branches:
    for br2 in branches:
        for k in (bz_mesh):
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                Q[br1,br2][time1,time2,k] = -delta[br1,br2][time1,time2] + epsg_impeps[br1,br2][time1,time2,k] - C[br1,br2][time1,time2,k]

print('Q is Hermitian:',is_hermitian(Q))

exit()
#eps_k = from_lesser_greater(eps_k_l, eps_k_l) #!

print(eps_k_test[FW,FW].data.shape)

print('esp_k_spin:',eps_k_test[FW,BW].data[0,0,0,:,:])
print('esp_k_spin:',eps_k_test[FW,BW].data[1,1,0,:,:])
print('esp_k_spin:',eps_k_test[FW,BW].data[0,0,1,:,:])
print('esp_k_spin:',eps_k_test[FW,BW].data[1,1,1,:,:])

print('is Hermitian:',is_hermitian(eps_k_test))
exit()
print(eps_k==eps_k_test)
#print('eps_k: ', eps_k[FW,FW].data.shape)
print('delta: ', delta[FW,FW].data.shape)
#print(eps_k.non_time_mesh.components)
#print(dir(eps_k.non_time_mesh))

for br1 in branches:
    for br2 in branches:
        eps_tild_k[br1,br2].data[...] = eps_k[br1,br2].data - delta[br1,br2].data[:,:,None,:,:]

for br1 in branches:
    for br2 in branches:
        for k in (bz_mesh):
            gf_imp_on_k_mesh[br1,br2].data[:,:,k.linear_index,:,:] = gf_imp[br1,br2].data[:,:,:,:]


#eps_tildgf_imp_conv = eps_tild_k @ gf_imp
F = eps_tild_k @ gf_imp

#for sp1 in (0,1):
#    for sp2 in (0,1):
#        for sp3 in (0,1):
#            print(sp1)
#            for br1 in branches:
#                for br2 in branches:
#                    eps_tild_gf_imp_vareps[br1,br2].data[:,:,:,sp1,sp2] \
#                     = eps_tildgf_imp_conv[br1,br2].data[:,:,:,sp1,sp3] \
#                                   * eps_k[br1,br2].data[:,:,:,sp3,sp2]


for sp in (0,1):
    print(sp)
    for br1 in branches:
        for br2 in branches:
            eps_tild_gf_imp_vareps[br1,br2].data[...] \
                        = F[br1,br2].data[:,:,:,:,None,sp] \
                          * eps_k[br1,br2].data[:,:,:,sp,None,:]

Q = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))

for br1 in branches:
    for br2 in branches:
        Q[br1,br2].data[...] = -delta[br1,br2].data[:,:,None,:,:] + eps_tild_gf_imp_vareps[br1,br2].data


#G = solve_vie2(F, Q)
#G = solve_vie2(F, F)
G = solve_vie2(F, eps_k_test)
#G = solve_vie2(F,eps_tild_gf_imp_vareps)
exit()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#for k in (bz_mesh):
#    print(k.linear_index)
#print(dir(bz_mesh))

print(eps_tild_k.arg_index_shapes)
print(gf_imp_on_k_mesh.arg_index_shapes)

eps_tild_k @ gf_imp_on_k_mesh






####
exit()

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

tri_vertex_test = compute_keldysh_vertex3(('dn', 0), # c_indices
                                        ('dn', 0), # c_dag_indices
                                        c_dag('dn', 0)*c_dag('up', 0), # n_op
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)

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


eps_k_l = Gf(mesh=ttk_mesh, target_shape=(2, 2)) #TODO: change to KeldyshGF 
#g_k_= KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
eps_tild_gf_imp_vareps = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))

print(eps_k_l.data[0,1,:,1,1])
for k in (bz_mesh):
    #print(k)
    for time1, time2 in MeshProduct(t_mesh, t_mesh):
        if time1.value == time2.value:
            eps_k_l[time1, time2, k] = eps_k(time1,time2,k)

#print(Q_l.data.shape)
#print(Q_l.data[0,0,:,0,0])
#print(Q_l.data[0,0,:,1,0])
#print(Q_l.data[0,0,:,0,1])
#print(Q_l.data[0,0,:,1,1])
#exit()
#print(Q_l.data[0,1,:,0,0])
#print(Q_l.data[1,1,:,0,0])
#print(Q_l)
eps_k = from_lesser_greater(eps_k_l, eps_k_l)

print('eps_k: ', eps_k[FW,FW].data.shape)
print('delta: ', delta[FW,FW].data.shape)

for br1 in branches:
    for br2 in branches:
        eps_tild_k[br1,br2].data[...] = eps_k[br1,br2].data - delta[br1,br2].data[:,:,None,:,:]

eps_tildgf_imp_conv = eps_tild_k @ gf_imp

exit()
