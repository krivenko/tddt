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
U = 6.0
Uch = U/2
Usp = -U/2
mu = 0.5 * U
eps = 0.0
t1 = 1.0 #0.25 # nearest neighbor hopping 
t2 = 0.0 # next nearest neighbor hopping
A = 0.0
Omega = 10
T = 0.0
A_x = 0.0
tx = 0.6*t1
occup_n = 0.5


# time-mesh
t_max = 2.0 # 10.0 #20.0
n_t = 31 #21
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


dt_pos_x = ti(m_interp, np.array([tx*np.exp(1.j * A_x * np.cos(Omega * x)) for x in m_interp]))
dt_neg_x = ti(m_interp, np.array([tx*np.exp(-1.j * A_x * np.cos(Omega * x)) for x in m_interp]))


# Lattice dispersion
eps_tk = Gf(mesh=tk_mesh, target_shape=(2, 2))

for t, k in tk_mesh:
    #eps_tk[t, k] = -2 * np.array([[[1, 0], [0, -1]]]) * np.cos(t.value) * (np.cos(k[0]) + np.cos(k[1]))
    eps_tk[t, k][0, 0] = 2.0*t1*(np.cos(k[0]-A*np.cos(Omega * t.value))+np.cos(k[1]-A*np.cos(Omega * t.value))) \
                         +4.0*t2*np.cos(2*(k[0]-A*np.cos(Omega * t.value)))*np.cos(2*(k[1]-A*np.cos(Omega * t.value)))
    eps_tk[t, k][1, 1] = 2.0*t1*(np.cos(k[0]-A*np.cos(Omega * t.value))+np.cos(k[1]-A*np.cos(Omega * t.value))) \
                         +4.0*t2*np.cos(2*(k[0]-A*np.cos(Omega * t.value)))*np.cos(2*(k[1]-A*np.cos(Omega * t.value)))

eps_loc = np.mean(eps_tk.data, axis = 1)

for t, k in tk_mesh:
    eps_tk[t, k] = eps_tk[t, k] - eps_loc[t.index, :]

eps_s2p_K = Singular2PKeldyshGF.from_retime(eps_tk)


# Dispersion reference system 
#TODO: compare to dt_pos/ dt_neg
def V(axis, sign, t):
    """ axis = x,y
        sign = -1,+1
    """
    V = tx * np.exp(sign * 1.j * A * np.cos(Omega * t))
    return V
for i in t_mesh:
    print(V('x', +1, i))

# Non-local interaction
def Vq(k, channel):
    Vq = 0.0
    return Vq


# Bare lattice interaction 
Uq = Gf(mesh=tk_mesh, target_shape=(2, 2))
Uq_tilde = Gf(mesh=tk_mesh, target_shape=(2, 2))

for t, k in tk_mesh:
    Uq[t, k][0,0] = Uch + Vq(k, 0) # charge
    Uq[t, k][1,1] = Usp + Vq(k, 1) # spin
    Uq_tilde[t, k][0,0] = Uq[t, k][0,0] - 0.5*Uch
    Uq_tilde[t, k][1,1] = Uq[t, k][1,1] - 0.5*Usp

Uq_s2p_K = Singular2PKeldyshGF.from_retime(Uq)
Uq_tilde_s2p_K = Singular2PKeldyshGF.from_retime(Uq_tilde)


# fermionic operators for the reference problem 
spin_names = ('up', 'dn')
fops = set(product(spin_names, [0, 1, 2, 3, 4]))


# Initial Hamiltonian #TODO: check which parameters to add


h0 = -mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0) \
     -0.0001 * (n('up', 1) + n('dn', 1)) \
     -0.0001 * (n('up', 2) + n('dn', 2)) \
     -0.0001 * (n('up', 3) + n('dn', 3)) \
     -0.0001 * (n('up', 4) + n('dn', 4)) \

h0 = h0 + \
    sum(dt_pos_x * c_dag(sn, 0) * c(sn, 1) + dt_neg_x * c_dag(sn, 1) * c(sn, 0)
        for sn in spin_names) + \
    sum(dt_pos_x * c_dag(sn, 0) * c(sn, 2) + dt_neg_x * c_dag(sn, 2) * c(sn, 0)
        for sn in spin_names) + \
    sum(dt_neg_x * c_dag(sn, 0) * c(sn, 3) + dt_pos_x * c_dag(sn, 3) * c(sn, 0)
        for sn in spin_names) + \
    sum(dt_neg_x * c_dag(sn, 0) * c(sn, 4) + dt_pos_x * c_dag(sn, 4) * c(sn, 0)
        for sn in spin_names)

init_state = make_equilibrium_init_state(h0,
                                         fermion_indices=fops,
                                         boson_indices=set(),
                                         temperature=T,
                                         params={})


# Hamiltonian after quench
#h = h0 + \
#    sum(dt_pos * c_dag(sn, 0) * c(sn, 1) + dt_neg * c_dag(sn, 1) * c(sn, 0)
#        for sn in spin_names) + \
#    sum(dt_pos * c_dag(sn, 0) * c(sn, 2) + dt_neg * c_dag(sn, 2) * c(sn, 0)
#        for sn in spin_names) + \
#    sum(dt_neg * c_dag(sn, 0) * c(sn, 3) + dt_pos * c_dag(sn, 3) * c(sn, 0)
#        for sn in spin_names) + \
#    sum(dt_neg * c_dag(sn, 0) * c(sn, 4) + dt_pos * c_dag(sn, 4) * c(sn, 0)
#        for sn in spin_names)

h = h0


params = {}
params['verbosity'] = 2                      # Verbosity level
params['hbar'] = 1.0                         # Planck's constant
params['hamiltonian_interpol'] = 'Trapezoid' # Trapezoid rule interpolation of H(t)
params['lanczos_min_matrix_size'] = 40       # Use LAPACK for subspaces of dim 32 and smaller



# Reference System GF
gf_struct_ref = [('up', 5), ('dn', 5)]
gf_ref = compute_keldysh_gf(gf_struct_ref,
                        init_state,
                        h,
                        t_mesh,
                        params)


# Make Keldysh GF of a single fermion with energy `eps` and occupation number `occup_n`
def make_g(eps, occup_n, t_mesh):
    tt_mesh = MeshProduct(t_mesh, t_mesh)
    g_l = Gf(mesh=tt_mesh, target_shape=(1, 1))
    g_g = Gf(mesh=tt_mesh, target_shape=(1, 1))
    for time1, time2 in tt_mesh:
        g_g[time1, time2] = -1j * (1.0 - occup_n) * np.exp(-1j * eps * (time1 - time2))
        g_l[time1, time2] = -1j * (-occup_n) * np.exp(-1j * eps * (time1 - time2))
    return KeldyshGF.from_lesser_greater(g_l, g_g)


# Bare impurity GF 
gimp0 = make_g(eps, occup_n, t_mesh)
print(gimp0.arg_index_shapes)

gimp = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,))) # GF for site 0 of the reference system
gref = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,5), (2,5))) # time, time, spin, site, spin, site for example: (11, 11, 2, 5, 2, 5)


for br1 in branches:
    for br2 in branches:
        gimp[br1,br2][0,0].data[...] = gf_ref['up'][br1,br2].data[...,0,0]
        gimp[br1,br2][1,1].data[...] = gf_ref['dn'][br1,br2].data[...,0,0]
        for i in range(5): # sites
            gref[br1,br2].data[...,0,i,0,i] = gf_ref['up'][br1,br2].data[...,i,i]
            gref[br1,br2].data[...,1,i,1,i] = gf_ref['dn'][br1,br2].data[...,i,i]

with open('data/tddt_ref_sys_t0_loc.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gf_ref['up'][FW,FW].data[0,t.index,0,0].real, gf_ref['up'][FW,FW].data[0,t.index,0,0].imag, 
                                          gf_ref['up'][FW,BW].data[0,t.index,0,0].real, gf_ref['up'][FW,BW].data[0,t.index,0,0].imag))

with open('data/tddt_ref_sys_01.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(gf_ref['up'][FW,FW].data[0,t.index,0,1].real, gf_ref['up'][FW,FW].data[0,t.index,0,1].imag, 
                                          gf_ref['up'][FW,BW].data[0,t.index,0,1].real, gf_ref['up'][FW,BW].data[0,t.index,0,1].imag))
        

# Hybridisation function delta
delta = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))

for sp in (0,1):
    for br1 in branches:
        for br2 in branches:
            for i in range(1,5):
                for time1, time2 in MeshProduct(t_mesh, t_mesh):
                    z1 = ContourPoint(br1, time1)
                    z2 = ContourPoint(br2, time2)
                    print('V: ', V('x',+1,time1.linear_index))
                    print('gimp0: ', gimp0[z1,z2][0,0])
                    delta[z1,z2][sp,sp] = V('x',+1,time1.linear_index) * gimp0[z1,z2][0,0] * V('x',-1,time2.linear_index) \
                                          + V('y',+1,time1.linear_index) * gimp0[z1,z2][0,0] * V('y',-1,time2.linear_index) \
                                          + V('x',-1,time1.linear_index) * gimp0[z1,z2][0,0] * V('x',+1,time2.linear_index) \
                                          + V('y',-1,time1.linear_index) * gimp0[z1,z2][0,0] * V('y',+1,time2.linear_index)


eps_gimp = eps_s2p_K @ gimp
eps_gimp_eps = eps_gimp @ eps_s2p_K
gimp_eps = gimp @ eps_s2p_K

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
                Q[br1,br2][time1,time2,k] = - delta[br1,br2][time1,time2] \
                                            + eps_gimp_eps[br1,br2][time1,time2,k] \
                                            - delta_gimp_eps[br1,br2][time1,time2,k]



Q = 0.5 * (Q + herm_conj(Q)) # for now to circumvent the hermicity check !!!!

Q_herm = herm_conj(Q)

F = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                F[br1,br2][time1,time2,k] = - eps_gimp[br1,br2][time1,time2,k] \
                                            + delta_gimp[br1,br2][time1,time2]


#print('F is Hermitian:', F.is_hermitian())

F = 0.5 * (F + herm_conj(F)) # for now to circumvent the check !!!!
Gd0_reg = solve_vie2(F, Q)
del F


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

    g_el = compute_keldysh_vertex3(c_index, # c_indices
                                   c_dag_index , # c_dag_indices
                                   n_op, # n_op
                                   init_state,
                                   h,
                                   t_mesh,
                                   params)
    return g_el


three_point_corr = KeldyshGF.from_arg_index_gen(generator_three_point_vertex, mesh=MeshProduct(t_mesh, t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)


U_pi_imp = KeldyshGF(mesh=MeshProduct(t_mesh, t_mesh), arg_index_shapes=((2,),(2,)))
for br1 in branches:
    for br2 in branches:
            U_pi_imp[br1,br2].data[:,:,0,0] = Uch*pi_imp[br1,br2].data[:,:,0,0]
            U_pi_imp[br1,br2].data[:,:,1,1] = Usp*pi_imp[br1,br2].data[:,:,1,1]

three_point_corr_U_pi_imp = conv(three_point_corr, U_pi_imp,
              [(2, 0)])


Lambda = three_point_corr - three_point_corr_U_pi_imp
del three_point_corr, three_point_corr_U_pi_imp


Uq_piimp = Uq_s2p_K @ pi_imp
Uq_piimp_Uq = Uq_piimp @ Uq_s2p_K

#Uq_piimp = 0.5 * (Uq_piimp + herm_conj(Uq_piimp)) # for now to circumvent the hermicity check !!!!
Uq_piimp_Uq = 0.5 * (Uq_piimp_Uq + herm_conj(Uq_piimp_Uq))
W0prime = solve_vie2(-Uq_piimp, Uq_piimp_Uq)

########################### DIAGRAMS #####################################

########################### prepare diagrams #####################################

#self-energy

eps_s2p_R = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
eps_s2p_mR = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
eps_s2p_loc = Singular2PKeldyshGF(mesh=t_mesh, arg_index_shapes=((2,), (2,)))


for time in t_mesh:
    for sigm in range(2):
        for br in branches:
            eps_s2p_loc[br][time][sigm,sigm] = np.mean(eps_s2p_K[br1][time,:][sigm,sigm].data)
            eps_K = eps_s2p_K[br][time,:][sigm,sigm].data.reshape(nkx,nky,nkz)

            eps_s2p_R[br].data[time.linear_index,:,sigm,sigm] = np.fft.ifftn(eps_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k -> R 
            eps_s2p_mR[br].data[time.linear_index,:,sigm,sigm] = np.fft.fftn(eps_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k+q -> mR 

print(eps_s2p_R[FW].data[0,:,0,0].reshape(nkx,nky,nkz))

k_points = list(bz_mesh)
q0 = k_points[0]


Uq_tilde_s2p_R = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
Uq0_tilde = Singular2PKeldyshGF(mesh=t_mesh, arg_index_shapes=((2,), (2,)))
for time in t_mesh:
    for ch in range(2):
        for br in branches:
            Uq_tilde_K = Uq_tilde_s2p_K[br][time,:][ch,ch].data.reshape(nkx,nky,nkz)
            Uq0_tilde[br][time][ch,ch] = Uq_tilde_s2p_K[br][time,q0][ch,ch]
            
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

########################### diagrams  prepare  END  #####################################

########################### Polarization #####################

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



Pi_R = -1j*(Pi_R_1 + Pi_R_2 + Pi_R_3 + Pi_R_4)

del LambdaGd0_reg_R, Lambdaeps_s2p_R, Gd0_reg_mRLambda, eps_s2p_mRLambda
del Pi_R_1, Pi_R_2, Pi_R_3, Pi_R_4

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

mFW = Uq_tilde_s2p_K_Pi_K
mQW = mFW @ Uq_tilde_s2p_K
QW = W0prime - mQW 

QW = -0.5 * (QW + herm_conj(QW)) # for now to circumvent the hermicity check !!!!
mFW = 0.5 * (mFW + herm_conj(mFW)) # for now to circumvent the hermicity check !!!!

Wprime = solve_vie2(-mFW, QW)

del mFW, mQW, QW, W0prime_Pi_K, Uq_tilde_s2p_K_Pi_K


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

LambdaGd0_reg_mR = conv(Lambda, Gd0_reg_mR,
             [(1, 0)])

Wprime_RLambda = conv(Wprime_R, Lambda,
             [(1, 2)])

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
del sigma_R_1, sigma_R_2, sigma_R_3,sigma_R_4

sigma_dual_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                sigma = sigma_R[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
                sigma_dual_K[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(sigma, axes=(0,1,2)).reshape(nkx*nky*nkz) # R -> K 

del sigma_R 

############# Tadpole #####################

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
                    sigma_dual_full[br1,br2][time1,time2,k][sigm,sigm] = sigma_dual_K[br1,br2][time1,time2,k][sigm,sigm] \
                                                                         + 0.0 * sigma_tadpole[br1,br2][time1,time2][sigm,sigm] #!!!!! tadpole is set to zero !!!!!

del sigma_dual_K, sigma_tadpole
############# Full Self-Energy END #####################

########################### Self-Energy END #####################################

K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                K[br1,br2][time1,time2,k] = sigma_dual_full[br1,br2][time1,time2,k][:,:] \
                                                       + gref[br1,br2][time1,time2][:,0,:,0]

gimp0[br1,br2][time1,time2][:,:]
L = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                L[br1,br2][time1,time2,k] = gref[br1,br2][time1,time2][:,0,:,0]


K_test = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
for br1 in branches:
    for br2 in branches:
        for k in bz_mesh:
            for time1, time2 in MeshProduct(t_mesh, t_mesh):
                K_test[br1,br2][time1,time2] = gref[br1,br2][time1,time2][:,0,:,0]
                        

Keps = K @ eps_s2p_K
Leps = L @ eps_s2p_K

Kdelta = K @ delta      
Ldelta = L @ delta   

FG = Kdelta - Keps
FG = 0.5 * (FG + herm_conj(FG)) # for now to circumvent the hermicity check !!!!
LG = Ldelta - Leps
LG = 0.5 * (LG + herm_conj(LG)) # for now to circumvent the hermicity check !!!!

K = 0.5 * (K + herm_conj(K)) # for now to circumvent the hermicity check !!!!
G_latt = solve_vie2(FG, K)
G_latt_CPT = solve_vie2(LG, L) # check because sigma_dual is in F!!!!!

del Keps, Kdelta, FG, L #, K


Gd0_K_full = K_test @ Gd0_reg + K_test @ eps_s2p_K
Gd0_K_full = Gd0_K_full @ K_test


G_latt_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
G_latt_CPT_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
G_latt_mR = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Gd0_full_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1 in branches:
        for br2 in branches:
            for sigm in range(2): # spin up and dn
                GR = G_latt[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
                GR_CPT  = G_latt_CPT[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)

                G_latt_CPT_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(GR_CPT, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R 
                
                G_latt_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(GR, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R                 
                G_latt_mR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.fftn(GR, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> -R 

                Gd0_full_K = Gd0_K_full[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
                Gd0_full_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(Gd0_full_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R 


with open('data/tddt_Gd0_R.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(Gd0_full_R[FW,FW].data[0,t.index,1,0,0].real, Gd0_full_R[FW,FW].data[0,t.index,1,0,0].imag, Gd0_full_R[FW,BW].data[0,t.index,1,0,0].real, Gd0_full_R[FW,BW].data[0,t.index,1,0,0].imag))


with open('data/tddt_t0_loc.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(np.sum(G_latt[FW,FW].data[0,t.index,:,0,0].real)/4.0, np.sum(G_latt[FW,FW].data[0,t.index,:,0,0].imag)/4.0, 
                                          np.sum(G_latt[FW,BW].data[0,t.index,:,0,0].real)/4.0, np.sum(G_latt[FW,BW].data[0,t.index,:,0,0].imag)/4.0))

with open('data/tddt_CPT.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(np.sum(G_latt_CPT[FW,FW].data[0,t.index,:,0].real)/4.0, np.sum(G_latt_CPT[FW,FW].data[0,t.index,:,0].imag)/4.0, 
                                          np.sum(G_latt_CPT[FW,BW].data[0,t.index,:,0].real)/4.0, np.sum(G_latt_CPT[FW,BW].data[0,t.index,:,0].imag)/4.0))

with open('data/K.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(np.sum(K[FW,FW].data[0,t.index,:,0,0].real)/4.0, np.sum(K[FW,FW].data[0,t.index,:,0,0].imag)/4.0, 
                                          np.sum(K[FW,BW].data[0,t.index,:,0,0].real)/4.0, np.sum(K[FW,BW].data[0,t.index,:,0,0].imag)/4.0))

with open('data/sigma_dual.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(np.sum(sigma_dual_full[FW,FW].data[0,t.index,:,0,0].real)/4.0, np.sum(sigma_dual_full[FW,FW].data[0,t.index,:,0,0].imag)/4.0, 
                                          np.sum(sigma_dual_full[FW,BW].data[0,t.index,:,0,0].real)/4.0, np.sum(sigma_dual_full[FW,BW].data[0,t.index,:,0,0].imag)/4.0))

with open('data/tddt_CPT_k0.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt_CPT[FW,FW].data[0,t.index,0,0,0].real, G_latt_CPT[FW,FW].data[0,t.index,0,0,0].imag, G_latt_CPT[FW,BW].data[0,t.index,0,0,0].real, G_latt_CPT[FW,BW].data[0,t.index,0,0,0].imag))

with open('data/tddt_CPT_k1.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt_CPT[FW,FW].data[0,t.index,1,0,0].real, G_latt_CPT[FW,FW].data[0,t.index,1,0,0].imag, G_latt_CPT[FW,BW].data[0,t.index,1,0,0].real, G_latt_CPT[FW,BW].data[0,t.index,1,0,0].imag))

with open('data/tddt_CPT_k2.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt_CPT[FW,FW].data[0,t.index,2,0,0].real, G_latt_CPT[FW,FW].data[0,t.index,2,0,0].imag, G_latt_CPT[FW,BW].data[0,t.index,2,0,0].real, G_latt_CPT[FW,BW].data[0,t.index,2,0,0].imag))

with open('data/tddt_CPT_k3.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt_CPT[FW,FW].data[0,t.index,3,0,0].real, G_latt_CPT[FW,FW].data[0,t.index,3,0,0].imag, G_latt_CPT[FW,BW].data[0,t.index,3,0,0].real, G_latt_CPT[FW,BW].data[0,t.index,3,0,0].imag))



with open('data/tddt_01.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(G_latt_R[FW,FW].data[0,t.index,1,0,0].real, G_latt_R[FW,FW].data[0,t.index,1,0,0].imag, 
                                          G_latt_R[FW,BW].data[0,t.index,1,0,0].real, G_latt_R[FW,BW].data[0,t.index,1,0,0].imag))

with open('data/tddt_CPT_01.txt', 'w') as file:
    # Loop to generate data
    file.write(f"# (FW,BW).local (FW,BW).local)\n")
    for t in t_mesh:
        # Write data to the first and second columnu
        file.write("{} {} {} {}\n".format(G_latt_CPT_R[FW,FW].data[0,t.index,1,0,0].real, G_latt_CPT_R[FW,FW].data[0,t.index,1,0,0].imag, 
                                          G_latt_CPT_R[FW,BW].data[0,t.index,1,0,0].real, G_latt_CPT_R[FW,BW].data[0,t.index,1,0,0].imag))


with open('data/tddt_T_t0_k0.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt[FW,FW].data[0,t.index,0,0,0].real, G_latt[FW,FW].data[0,t.index,0,0,0].imag, G_latt[FW,BW].data[0,t.index,0,0,0].real, G_latt[FW,BW].data[0,t.index,0,0,0].imag))

with open('data/tddt_T_t0_k1.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt[FW,FW].data[0,t.index,1,0,0].real, G_latt[FW,FW].data[0,t.index,1,0,0].imag, G_latt[FW,BW].data[0,t.index,1,0,0].real, G_latt[FW,BW].data[0,t.index,1,0,0].imag))

with open('data/tddt_T_t0_k2.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt[FW,FW].data[0,t.index,2,0,0].real, G_latt[FW,FW].data[0,t.index,2,0,0].imag, G_latt[FW,BW].data[0,t.index,2,0,0].real, G_latt[FW,BW].data[0,t.index,2,0,0].imag))

with open('data/tddt_T_t0_k3.txt', 'w') as file:
    # Loop to generate data
    for t in t_mesh:
        # Write data to the first and second columns
        file.write("{} {} {} {}\n".format(G_latt[FW,FW].data[0,t.index,3,0,0].real, G_latt[FW,FW].data[0,t.index,3,0,0].imag, G_latt[FW,BW].data[0,t.index,3,0,0].real, G_latt[FW,BW].data[0,t.index,3,0,0].imag))
