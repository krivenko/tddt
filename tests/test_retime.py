import unittest
from itertools import product
import numpy as np

from triqs.gf import MeshReTime, MeshBrillouinZone, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone
from triqs.utility.comparison_tests import assert_gfs_are_close

from tddt.retime import conj, conv_ret_ret, conv_ret_lg, conv_lg_adv
from tddt.keldysh import (from_lesser_greater,
                          retarded,
                          retarded_ext,
                          advanced,
                          advanced_ext)
from tddt.integration import GregoryIntegrator


class test_retime(unittest.TestCase):
    """Auxiliary routines for Green's functions of real time"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = 20.0
        cls.n_t = 11
        cls.t_mesh = MeshReTime(0, cls.t_max, cls.n_t)

        cls.bl = BravaisLattice(units=[(1, 0, 0)])  # Square lattice

    def _make_g_l_g_g_scalar(self, x=1.0):
        mesh = MeshProduct(self.t_mesh, self.t_mesh)
        g_l = Gf(mesh=mesh, target_shape=())
        g_g = Gf(mesh=mesh, target_shape=())
        for t1, t2 in mesh:
            e = np.exp(-1j * (t1.value - t2.value))
            g_g[t1, t2] = -1j * (1.0 - 0.1 * x) * e
            g_l[t1, t2] = -1j * (-0.1 * x) * e
        return g_l, g_g

    def _make_g_l_g_g_matrix(self, x=1.0):
        mesh = MeshProduct(self.t_mesh, self.t_mesh)
        g_l = Gf(mesh=mesh, target_shape=(3, 3))
        g_g = Gf(mesh=mesh, target_shape=(3, 3))
        H = np.array([[1.0, 0.5, -0.1j],
                      [0.5, 2.0, 0.5],
                      [0.1j, 0.5, 3.0]])
        E, U = np.linalg.eig(H)
        for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
            dt = (t1.value - t2.value)
            e = U @ np.diag(np.exp(-1j * x * E * dt)) @ np.conj(U.T)
            g_g[t1, t2] = -1j * (1.0 - 0.1 * x) * e
            g_l[t1, t2] = -1j * (-0.1 * x) * e
        return g_l, g_g

    def _make_g_l_g_g_matrix_bz(self, bz_mesh, x=1.0):
        mesh = MeshProduct(self.t_mesh, self.t_mesh, bz_mesh)

        eps_k = np.array([k.value[0] / np.pi + 1.0 for k in bz_mesh])

        g_l = Gf(mesh=mesh, target_shape=(3, 3))
        g_g = Gf(mesh=mesh, target_shape=(3, 3))
        for k, eps in zip(bz_mesh, eps_k):
            H = np.array([[eps, 0.5, -0.1j],
                          [0.5, 2.0 * eps, 0.5],
                          [0.1j, 0.5, 3.0 * eps]])
            E, U = np.linalg.eig(H)
            for t1, t2 in MeshProduct(self.t_mesh, self.t_mesh):
                dt = (t1.value - t2.value)
                e = U @ np.diag(np.exp(-1j * x * E * dt)) @ np.conj(U.T)
                g_g[t1, t2, k] = -1j * (1.0 - 0.1 * x) * e
                g_l[t1, t2, k] = -1j * (-0.1 * x) * e
        return g_l, g_g

    def test_conj(self):
        # Scalar-valued GF
        g_l, g_g = self._make_g_l_g_g_scalar()
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_gfs_are_close(g_adv, g_adv_ref, precision=1e-12)

        # Matrix-valued GF
        g_l, g_g = self._make_g_l_g_g_matrix()
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_gfs_are_close(g_adv, g_adv_ref, precision=1e-12)

        # Matrix-valued GF with an extra mesh component
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), 5)
        g_l, g_g = self._make_g_l_g_g_matrix_bz(bz_mesh)
        g = from_lesser_greater(g_l, g_g)

        g_adv = conj(retarded(g))
        g_adv_ref = advanced(g)
        assert_gfs_are_close(g_adv, g_adv_ref, precision=1e-12)

    def test_conv_scalar(self):
        order = 5
        integrator = GregoryIntegrator(order)
        w = integrator.weights(self.t_mesh)

        # Scalar-valued GF
        a_l, a_g = self._make_g_l_g_g_scalar(1.0)
        b_l, b_g = self._make_g_l_g_g_scalar(2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_ret_ext = retarded_ext(b)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_lg(a_ret_ext, b_l)
        a_l_b_adv = conv_lg_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=())
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=())
        for n, m in product(range(self.n_t), range(self.n_t)):
            a_ret_b_l_ref.data[n, m] = sum(
                w[n, j] * a_ret_ext.data[n, j] * b_l.data[j, m]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m] = sum(
                w[m, j] * a_l.data[n, j] * b_adv_ext.data[j, m]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)

        a_ret_b_ret = conv_ret_ret(a_ret_ext, b_ret_ext)

        a_ret_b_ret_ref = Gf(mesh=a_ret_b_ret.mesh, target_shape=())
        for n, m in product(range(self.n_t), range(self.n_t)):
            if n < m:
                continue
            if n <= order:
                a_ret_b_ret_ref.data[n, m] = self.t_mesh.delta * sum(
                    integrator.I[m, n, j]
                    * a_ret_ext.data[n, j] * b_ret_ext.data[j, m]
                    for j in range(order + 1)
                )
            elif n - m > order:
                a_ret_b_ret_ref.data[n, m] = sum(
                    w[n - m, j - m]
                    * a_ret_ext.data[n, j] * b_ret_ext.data[j, m]
                    for j in range(m, n + 1)
                )
            else:
                a_ret_b_ret_ref.data[n, m] = sum(
                    w[n - m, j]
                    * a_ret_ext.data[n, n - j] * b_ret_ext.data[n - j, m]
                    for j in range(order + 1)
                )

        assert_gfs_are_close(a_ret_b_ret, a_ret_b_ret_ref, precision=1e-12)

    def test_conv_matrix(self):
        order = 5
        integrator = GregoryIntegrator(order)
        w = integrator.weights(self.t_mesh)

        # Matrix-valued GF
        a_l, a_g = self._make_g_l_g_g_matrix(1.0)
        b_l, b_g = self._make_g_l_g_g_matrix(2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_ret_ext = retarded_ext(b)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_lg(a_ret_ext, b_l)
        a_l_b_adv = conv_lg_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=(3, 3))
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=(3, 3))
        for n, m, a, b, c in product(range(self.n_t),
                                     range(self.n_t),
                                     *[range(3)] * 3):
            a_ret_b_l_ref.data[n, m, a, b] += sum(
                w[n, j] * a_ret_ext.data[n, j, a, c] * b_l.data[j, m, c, b]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m, a, b] += sum(
                w[m, j] * a_l.data[n, j, a, c] * b_adv_ext.data[j, m, c, b]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)

        a_ret_b_ret = conv_ret_ret(a_ret_ext, b_ret_ext)

        a_ret_b_ret_ref = Gf(mesh=a_ret_b_ret.mesh, target_shape=(3, 3))
        for n, m, a, b, c in product(range(self.n_t),
                                     range(self.n_t),
                                     *[range(3)] * 3):
            if n < m:
                continue
            if n <= order:
                a_ret_b_ret_ref.data[n, m, a, b] += self.t_mesh.delta * sum(
                    integrator.I[m, n, j]
                    * a_ret_ext.data[n, j, a, c] * b_ret_ext.data[j, m, c, b]
                    for j in range(order + 1)
                )
            elif n - m > order:
                a_ret_b_ret_ref.data[n, m, a, b] += sum(
                    w[n - m, j - m]
                    * a_ret_ext.data[n, j, a, c] * b_ret_ext.data[j, m, c, b]
                    for j in range(m, n + 1)
                )
            else:
                a_ret_b_ret_ref.data[n, m, a, b] += sum(
                    w[n - m, j]
                    * a_ret_ext.data[n, n - j, a, c]
                    * b_ret_ext.data[n - j, m, c, b]
                    for j in range(order + 1)
                )

        assert_gfs_are_close(a_ret_b_ret, a_ret_b_ret_ref, precision=1e-12)

    def test_conv_matrix_bz(self):
        order = 5
        integrator = GregoryIntegrator(order)
        w = integrator.weights(self.t_mesh)

        # Matrix-valued GF with an extra mesh component
        n_k = 5
        bz_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k)

        a_l, a_g = self._make_g_l_g_g_matrix_bz(bz_mesh, 1.0)
        b_l, b_g = self._make_g_l_g_g_matrix_bz(bz_mesh, 2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_ret_ext = retarded_ext(b)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_lg(a_ret_ext, b_l)
        a_l_b_adv = conv_lg_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=(3, 3))
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=(3, 3))
        for n, m, K, a, b, c in product(range(self.n_t),
                                        range(self.n_t),
                                        range(n_k),
                                        *[range(3)] * 3):
            a_ret_b_l_ref.data[n, m, K, a, b] += sum(
                w[n, j]
                * a_ret_ext.data[n, j, K, a, c] * b_l.data[j, m, K, c, b]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m, K, a, b] += sum(
                w[m, j]
                * a_l.data[n, j, K, a, c] * b_adv_ext.data[j, m, K, c, b]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)

        a_ret_b_ret = conv_ret_ret(a_ret_ext, b_ret_ext)

        a_ret_b_ret_ref = Gf(mesh=a_ret_b_ret.mesh, target_shape=(3, 3))
        for n, m, K, a, b, c in product(range(self.n_t),
                                        range(self.n_t),
                                        range(n_k),
                                        *[range(3)] * 3):
            if n < m:
                continue
            if n <= order:
                a_ret_b_ret_ref.data[n, m, K, a, b] += self.t_mesh.delta * sum(
                    integrator.I[m, n, j]
                    * a_ret_ext.data[n, j, K, a, c]
                    * b_ret_ext.data[j, m, K, c, b]
                    for j in range(order + 1)
                )
            elif n - m > order:
                a_ret_b_ret_ref.data[n, m, K, a, b] += sum(
                    w[n - m, j - m]
                    * a_ret_ext.data[n, j, K, a, c]
                    * b_ret_ext.data[j, m, K, c, b]
                    for j in range(m, n + 1)
                )
            else:
                a_ret_b_ret_ref.data[n, m, K, a, b] += sum(
                    w[n - m, j]
                    * a_ret_ext.data[n, n - j, K, a, c]
                    * b_ret_ext.data[n - j, m, K, c, b]
                    for j in range(order + 1)
                )

        assert_gfs_are_close(a_ret_b_ret, a_ret_b_ret_ref, precision=1e-12)

    def test_conv_matrix_bz1_bz2(self):
        order = 5
        integrator = GregoryIntegrator(order)
        w = integrator.weights(self.t_mesh)

        # Matrix-valued GF with an extra mesh component
        n_k1 = 4
        bz1_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k1)
        n_k2 = 3
        bz2_mesh = MeshBrillouinZone(BrillouinZone(self.bl), n_k2)

        a_l, a_g = self._make_g_l_g_g_matrix_bz(bz1_mesh, 1.0)
        b_l, b_g = self._make_g_l_g_g_matrix_bz(bz2_mesh, 2.0)
        a = from_lesser_greater(a_l, a_g)
        b = from_lesser_greater(b_l, b_g)

        a_ret_ext = retarded_ext(a)
        b_ret_ext = retarded_ext(b)
        b_adv_ext = advanced_ext(b)

        a_ret_b_l = conv_ret_lg(a_ret_ext, b_l)
        a_l_b_adv = conv_lg_adv(a_l, b_adv_ext)

        a_ret_b_l_ref = Gf(mesh=a_ret_b_l.mesh, target_shape=(3, 3))
        a_l_b_adv_ref = Gf(mesh=a_l_b_adv.mesh, target_shape=(3, 3))
        for n, m, K1, K2, a, b, c in product(range(self.n_t),
                                             range(self.n_t),
                                             range(n_k1),
                                             range(n_k2),
                                             *[range(3)] * 3):
            a_ret_b_l_ref.data[n, m, K1, K2, a, b] += sum(
                w[n, j]
                * a_ret_ext.data[n, j, K1, a, c] * b_l.data[j, m, K2, c, b]
                for j in range((n if (n > order) else order) + 1)
            )
            a_l_b_adv_ref.data[n, m, K1, K2, a, b] += sum(
                w[m, j]
                * a_l.data[n, j, K1, a, c] * b_adv_ext.data[j, m, K2, c, b]
                for j in range((m if (m > order) else order) + 1)
            )

        assert_gfs_are_close(a_ret_b_l, a_ret_b_l_ref, precision=1e-12)
        assert_gfs_are_close(a_l_b_adv, a_l_b_adv_ref, precision=1e-12)

        a_ret_b_ret = conv_ret_ret(a_ret_ext, b_ret_ext)

        a_ret_b_ret_ref = Gf(mesh=a_ret_b_ret.mesh, target_shape=(3, 3))
        for n, m, K1, K2, a, b, c in product(range(self.n_t),
                                             range(self.n_t),
                                             range(n_k1),
                                             range(n_k2),
                                             *[range(3)] * 3):
            if n < m:
                continue
            if n <= order:
                a_ret_b_ret_ref.data[n, m, K1, K2, a, b] += \
                    self.t_mesh.delta * sum(
                    integrator.I[m, n, j]
                    * a_ret_ext.data[n, j, K1, a, c]
                    * b_ret_ext.data[j, m, K2, c, b]
                    for j in range(order + 1)
                )
            elif n - m > order:
                a_ret_b_ret_ref.data[n, m, K1, K2, a, b] += sum(
                    w[n - m, j - m]
                    * a_ret_ext.data[n, j, K1, a, c]
                    * b_ret_ext.data[j, m, K2, c, b]
                    for j in range(m, n + 1)
                )
            else:
                a_ret_b_ret_ref.data[n, m, K1, K2, a, b] += sum(
                    w[n - m, j]
                    * a_ret_ext.data[n, n - j, K1, a, c]
                    * b_ret_ext.data[n - j, m, K2, c, b]
                    for j in range(order + 1)
                )

        assert_gfs_are_close(a_ret_b_ret, a_ret_b_ret_ref, precision=1e-12)


if __name__ == '__main__':
    unittest.main()
