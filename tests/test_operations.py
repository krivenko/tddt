import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime

from realevol.tinterp import TInterp as ti
from realevol.operators_tinterp import *
from realevol.init_state import *

from tddt.realevol import compute_keldysh_conn_correlator_2t

from tddt.operations import invert_keldysh_correlator_2t

# FIXME
from tddt.keldysh import Branch

class test_operations(unittest.TestCase):
    """Operations with correlators on Keldysh contour"""

    @classmethod
    def setUpClass(cls):
        cls.spin_names = ('up', 'dn')
        cls.t_max = 5.0
        cls.n_t = 5
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)

        m_interp = MeshReTime(0, cls.t_max, 1001)

        # Model parameters
        U = 3.0
        mu = 0.5 * U
        eps = 0.2
        t = 0.3
        dt = ti(m_interp, np.array([0.1*(1-np.exp(-5*x)) for x in m_interp]))

        fops = set(product(cls.spin_names,[0,1]))

        # Initial Hamiltonian
        h0 = -mu*(n('up',0) + n('dn',0)) + U * n('up',0) * n('dn',0)
        h0 += eps*(n('up',1) + n('dn',1))
        h0 += sum(-t*(c_dag(sn,0) * c(sn,1) + c_dag(sn,1) * c(sn,0))
                  for sn in cls.spin_names)

        cls.init_state = make_equilibrium_init_state(h0,
                                                     fermion_indices = fops,
                                                     boson_indices = set(),
                                                     temperature = 0,
                                                     params = {})

        # Hamiltonian after quench
        cls.h = h0 + sum(dt*(c_dag(sn,0)*c(sn,1) + c_dag(sn,1)*c(sn,0))
                         for sn in cls.spin_names)

        cls.params = {}
        cls.params['verbosity'] = 2
        cls.params['lanczos_min_matrix_size'] = 10000

    def test_invert_keldysh_correlator_2t(self):
        Sz = 0.5 * (n('up', 0) - n('dn', 0))
        SzSz = compute_keldysh_conn_correlator_2t(Sz, Sz,
                                                  self.init_state,
                                                  self.h,
                                                  self.t_mesh,
                                                  self.params)

        # FIXME
        #def save_data(name, g):
        #    np.savetxt(name + ".re.dat", g.data.real)
        #    np.savetxt(name + ".im.dat", g.data.imag)

        #save_data("SzSz.FF", SzSz[Branch.FORWARD, Branch.FORWARD])
        #save_data("SzSz.FB", SzSz[Branch.FORWARD, Branch.BACKWARD])
        #save_data("SzSz.BF", SzSz[Branch.BACKWARD, Branch.FORWARD])
        #save_data("SzSz.BB", SzSz[Branch.BACKWARD, Branch.BACKWARD])

        invert_keldysh_correlator_2t(SzSz)

