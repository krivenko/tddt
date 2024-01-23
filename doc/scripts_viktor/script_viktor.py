#from itertools import product
import numpy as np
from itertools import product

#import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, Gf
from triqs.gf import MeshProduct # A direct product of 1D meshes
import triqs.utility.mpi as mpi         # MPI utilities

#from realevol.tinterp import TInterp as ti
from realevol.operators_tinterp import c, c_dag, n
from realevol.init_state import make_equilibrium_init_state
from realevol.realevol import compute_expectval

from tddt.realevol import (
    compute_keldysh_gf,
    compute_keldysh_gf_element,
    compute_keldysh_correlator_2t,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)
#from tddt.keldysh import Branch, KeldyshGF, from_lesser_greater

from tddt.keldysh import  (KeldyshGF,
                           greater,
                           lesser,
                           retarded,
                           advanced)

#from tddt.realevol import (
#    compute_keldysh_gf,
#    compute_keldysh_gf_element,
#    compute_keldysh_correlator_2t,
#    compute_keldysh_conn_correlator_2t,
#    compute_keldysh_vertex3
#)
#from tddt.testing import assert_keldysh_gf_almost_equal

t_max = 5.0                                # Maximum observation time
n_t = 7                                    # Number of slices in the time grid
t_mesh = MeshReTime(0, t_max, n_t)         # A 1D real time grid
tt_mesh = MeshProduct(t_mesh, t_mesh)      # A 2D mesh as a direct product of t_mesh with itself
#m_interp = MeshReTime(0, t_max, 1001)     # Fine mesh to construct interpolation objects

# Model parameters
U = 3.0
mu = 0.5 * U
eps = 0.2
t = 0.3
#dt = ti(m_interp,
#        np.array([0.1 * (1 - np.exp(-5 * x)) for x in m_interp]))

spin_names = ('up', 'dn')
fops = set(product(spin_names, [0, 1]))

# Initial Hamiltonian
h0 = -mu * (n('up', 0) + n('dn', 0)) + U * n('up', 0) * n('dn', 0)
h0 += eps * (n('up', 1) + n('dn', 1))
h0 += sum(-t * (c_dag(sn, 0) * c(sn, 1) + c_dag(sn, 1) * c(sn, 0))
                for sn in spin_names)

init_state = make_equilibrium_init_state(h0,
                                         fermion_indices=fops,
                                         boson_indices=set(),
                                         temperature=0,
                                         params={})

h = h0

# Calculation parameters
params = {}
params['verbosity'] = 2                      # Verbosity level
params['hbar'] = 1.0                         # Planck's constant
params['hamiltonian_interpol'] = 'Trapezoid' # Trapezoid rule interpolation of H(t)
params['lanczos_min_matrix_size'] = 40       # Use LAPACK for subspaces of dim 32 and smaller

gf_struct = [('up', 2), ('dn', 2)]
g_matrix = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))

g_k_matrix = compute_keldysh_gf(gf_struct, init_state, h, t_mesh, params)

print(g_k_matrix)


