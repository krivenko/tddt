#
# Volterra integral equations of the 2nd kind
#

from itertools import product
import numpy as np

from triqs.gf import Gf, MeshReTime

from .retime import conj, conv_lg_adv
from .keldysh import (KeldyshGF,
                      lesser,
                      retarded,
                      retarded_ext,
                      is_hermitian,
                      from_lesser_greater)
from .integration import GregoryIntegrator


class VIE2Solver:
    r"""
    Solver for a few families of Volterra integral equations of the second kind.
    """

    def __init__(self, mesh: MeshReTime, shape, /, gregory_order=5):
        r"""
        mesh: Each real time argument of the integral kernels, right-hand sides
        and solutions takes values from this time grid.

        shape: Tensor shape associated with each real time argument.

        gregory_order: Order of the Gregory quadrature rule used to solve the
        equations.
        """

        self.N = len(mesh)
        assert self.N >= gregory_order + 1
        self.dt = mesh.delta

        self.shape = shape

        self.integrator = GregoryIntegrator(gregory_order)
        self.f1_shape = (self.N, *shape)
        self.f2_shape = (self.N, self.N, *shape, *shape)

        self.I = self.integrator.I  # noqa: E741
        self.w = self.integrator.weights(mesh)

        # Preallocate self.startup_mat with the maximal required size that
        # corresponds to lower_limit=0.
        s_shape = self._startup_shape(0)
        self.startup_mat = np.zeros(s_shape + s_shape, dtype=complex)
        self.startup_rhs = np.zeros(s_shape, dtype=complex)

        if shape:
            self.stepping_mat = np.zeros(shape + shape, dtype=complex)
            self.stepping_rhs = np.zeros(shape, dtype=complex)
        else:
            self.stepping_mat = np.zeros((1, 1), dtype=complex)
            self.stepping_rhs = np.zeros(1, dtype=complex)

    def _startup_shape(self, m: int):
        return (self.integrator.order - m, *self.shape)

    def _startup(self, k: np.ndarray, q: np.ndarray, y: np.ndarray):
        y[0, ...] = q[0, ...]

        order = self.integrator.order
        sol_slice = (slice(None),) * len(self.shape)
        s_shape = self._startup_shape(0)

        self.startup_mat[:] = np.eye(np.prod(s_shape)) \
            .reshape(self.startup_mat.shape)
        for n, l in product(range(1, order + 1), repeat=2):
            self.startup_mat[(n - 1, *sol_slice, l - 1, *sol_slice)] += \
                self.w[n, l] * k[n, l, ...]

        self.startup_rhs[:, ...] = q[1:(order + 1), ...]
        for n in range(1, order + 1):
            self.startup_rhs[n - 1, ...] -= \
                self.w[n, 0] * \
                np.tensordot(k[n, 0, ...], y[0, ...], axes=y.ndim - 1)

        y[1:(order + 1), ...] = \
            np.linalg.tensorsolve(self.startup_mat, self.startup_rhs)

    def _step(self, k: np.ndarray, q: np.ndarray, y: np.ndarray, n: int):
        self.stepping_mat[:] = np.eye(int(np.prod(self.shape))) \
            .reshape(self.stepping_mat.shape)
        self.stepping_mat += self.w[n, n] * k[n, n, ...]

        self.stepping_rhs[:] = q[n, ...]
        for l in range(n):  # noqa: E741
            self.stepping_rhs -= self.w[n, l] \
                * np.tensordot(k[n, l, ...], y[l, ...], axes=y.ndim - 1)

        y[n, ...] = np.linalg.tensorsolve(self.stepping_mat, self.stepping_rhs)

    def __call__(self, k: np.ndarray, q: np.ndarray):
        r"""
        Solve the Volterra integral equation of the second kind

            y_a(t) + \sum_b \int_0^t ds k_{a,b}(t, s) y_b(s) = q_a(t)

        w.r.t. y_a(t) on a uniform real time mesh using start-up and
        time-stepping procedures based on the Gregory quadrature rule. The
        solution y_a(t) is, in general, tensor-valued with an arbitrary shape
        and 'a' is the corresponding multi-index.

        The array `q` is the right hand side. Its first axis corresponds to the
        time argument and the rest of axes (if any) correspond to the
        multi-index 'a' with dimensions specified by the `shape` tuple
        passed to the constructor.

        The array `k` is the integral kernel. Its layout must be such that

        * Its first 2 axes correspond to the time arguments t and s.
        * The next `len(shape)` axes correspond to the multi-index 'a'.
        * The final `len(shape)` axes correspond to the multi-index 'b'.

        The returned solution y_a(t) has the same layout as `q`.
        """
        assert k.shape == self.f2_shape
        assert q.shape == self.f1_shape

        y = np.empty(self.f1_shape, dtype=complex)

        self._startup(k, q, y)
        for n in range(self.integrator.order + 1, self.N):
            self._step(k, q, y, n)

        return y

    def _herm_conj(self, a):
        ls = len(self.shape)
        hc_axes = (1, 0, *range(2 + ls, 2 + 2 * ls), *range(2, 2 + ls))
        return -np.conj(np.transpose(a, hc_axes))

    def _causal_startup(self,
                        f: np.ndarray,
                        q: np.ndarray,
                        g: np.ndarray,
                        m: int):
        order = self.integrator.order
        shape_sl = (slice(None),) * len(self.shape)

        # For the given m, make partial views of the preallocated
        # self.startup_mat and self.startup_rhs.
        s_shape = self._startup_shape(m)
        dim = s_shape[0]
        mat = self.startup_mat[(slice(dim), *shape_sl, slice(dim), *shape_sl)]
        rhs = self.startup_rhs[(slice(dim), *shape_sl)]

        # Iterate over the second multi-index of g and q
        for b in np.ndindex(self.shape):
            mat[:] = np.eye(np.prod(s_shape)).reshape(s_shape + s_shape)

            for n, l in product(range(m + 1, order + 1), repeat=2):
                mat[(n - (m + 1), *shape_sl, l - (m + 1), *shape_sl)] += \
                    self.dt * self.I[m, n, l] * f[n, l, ...]

            rhs[:] = q[(slice(m + 1, order + 1), m, *shape_sl, *b)]
            for n, l in product(range(m + 1, order + 1), range(m + 1)):
                rhs[n - (m + 1), ...] -= self.dt * self.I[m, n, l] * \
                    np.tensordot(f[n, l, ...],
                                 g[(l, m, *shape_sl, *b)],
                                 axes=len(self.shape))

            y = g[(slice(m + 1, order + 1), m, *shape_sl, *b)]
            y[:] = np.linalg.tensorsolve(mat, rhs)
            g[(m, slice(m + 1, order + 1), *b, *shape_sl)] = -np.conj(y)

    def _causal_step(self,
                     f: np.ndarray,
                     q: np.ndarray,
                     g: np.ndarray,
                     m: int,
                     n: int):
        shape_sl = (slice(None),) * len(self.shape)

        # Iterate over the second multi-index of g and q
        for b in np.ndindex(self.shape):
            self.stepping_mat[:] = \
                np.eye(int(np.prod(self.shape)), dtype=complex) \
                .reshape(self.stepping_mat.shape)

            self.stepping_rhs[:] = q[(n, m, *shape_sl, *b)]
            if n - m > self.integrator.order:
                self.stepping_mat += self.w[n - m, n - m] * f[n, n, ...]
                for j in range(m, n):
                    self.stepping_rhs -= self.w[n - m, j - m] \
                        * np.tensordot(f[n, j, ...],
                                       g[(j, m, *shape_sl, *b)],
                                       axes=len(self.shape))
            else:  # n - m <= self.integrator.order
                self.stepping_mat += self.w[n - m, 0] * f[n, n, ...]
                for j in range(1, self.integrator.order + 1):
                    self.stepping_rhs -= self.w[n - m, j] \
                        * np.tensordot(f[n, n - j, ...],
                                       g[(n - j, m, *shape_sl, *b)],
                                       axes=len(self.shape))

            g[(n, m, *shape_sl, *b)] = np.linalg.tensorsolve(self.stepping_mat,
                                                             self.stepping_rhs)
            g[(m, n, *b, *shape_sl)] = -np.conj(g[(n, m, *shape_sl, *b)])

    def solve_causal(self, f: np.ndarray, q: np.ndarray):
        r"""
        Solve the causal Volterra integral equation of the second kind

            g_{a,b}(t, t') + \sum_c \int_{t'}^t ds f_{a,c}(t, s) g_{c,b}(s, t')
                = q_{a,b}(t, t')

        w.r.t. g_{a,b}(t, t') on a uniform real time mesh using start-up and
        time-stepping procedures based on the Gregory quadrature rule. The
        solution g_{a, b}(t, t') is, in general, tensor-valued with 'a' and
        'b' being the corresponding multi-indices.

        Arrays 'f' and 'q' must have the following layouts.

        * The first 2 axes correspond to the time arguments.
        * The next `len(shape)` axes correspond to the first multi-index.
        * The final `len(shape)` axes correspond to the second multi-index.

        The kernel f_{a, b}(t, t') must obey the Hermitian symmetry property

            f_{a, b}(t, t') = -conj(f_{b, a}(t', t)).

        The returned solution g_{a,b}(t, t') has the same layout as `q`.
        """
        assert f.shape == self.f2_shape
        assert q.shape == self.f2_shape

        np.testing.assert_allclose(f, self._herm_conj(f), atol=1e-14)

        g = np.empty(q.shape, dtype=complex)

        # Copy the diagonal elements t = t' from q
        diag_idx = np.diag_indices(g.shape[0], ndim=2)
        g[diag_idx] = q[diag_idx]

        for m in range(self.integrator.order):
            self._causal_startup(f, q, g, m)
        for m in range(g.shape[1] - 1):
            for n in range(max(self.integrator.order, m) + 1, g.shape[0]):
                self._causal_step(f, q, g, m, n)

        return g


def solve_vie2(F: KeldyshGF, Q: KeldyshGF) -> KeldyshGF:
    r"""
    Solve the contour Dyson equation in integral form,

      G(t, t') + \int_C d\bar t F(t, \bar t) G(\bar t, t') = Q(t, t')

    w.r.t. G(t, t') with a Hermitian Q(t, t') and

        \int_C d\bar t F(t, \bar t) Q(\bar t, t') =
        \int_C d\bar t Q(t, \bar t) F^\dagger(\bar t, t').

    The resulting G(t, t') is also Hermitian.
    """
    assert is_hermitian(Q), "Q must be Hermitian"
    assert F.n_args == 2, "F must be a 2-point GF"
    assert F.time_mesh.components[1] == Q.time_mesh.components[0], \
        "Incompatible time meshes of F and Q"
    assert F.arg_index_shapes[1] == Q.arg_index_shapes[0], \
        "Incompatible target subshapes of F and Q"
    assert F.non_time_mesh == Q.non_time_mesh, \
        "Different non-time meshes of F and Q"
    assert F.time_mesh.components[0] == F.time_mesh.components[1], \
        "Time mesh of F must be square"
    assert F.arg_index_shapes[0] == F.arg_index_shapes[1], \
        "The two target subshapes of F must be equal"
    # FIXME: This check must be enabled as soon as keldysh.conv() is fixed
    # assert_keldysh_gf_almost_equal(
    #     F @ Q, Q @ herm_conj(F),
    #     1e-14,
    #     err_msg=r"F and Q must satisfy F * Q == Q * F^\ddagger"
    # )

    solver = VIE2Solver(Q.time_mesh.components[0],
                        Q.arg_index_shapes[0],
                        gregory_order=Q.integrator.order)

    non_time_mesh_shape = tuple(map(len, Q.non_time_mesh.components))

    #
    # Solve equation for the extended retarded component of G
    #

    Q_ret = retarded(Q)
    F_ret_ext = retarded_ext(F)
    G_ret_ext = Gf(mesh=Q.mesh, target_shape=Q.target_shape)

    for non_t_idx in np.ndindex(non_time_mesh_shape):
        s = (slice(None), slice(None), *non_t_idx, Ellipsis)
        G_ret_ext.data[s] = solver.solve_causal(F_ret_ext.data[s],
                                                Q_ret.data[s])

    #
    # Solve equation for the lesser component of G
    #

    F_l = lesser(F)
    G_adv_ext = conj(G_ret_ext)  # This applies only for a Hermitian G(t, t')
    rhs_l = lesser(Q) - conv_lg_adv(F_l, G_adv_ext)
    G_l = Gf(mesh=rhs_l.mesh, target_shape=rhs_l.target_shape)

    time_mesh_shape2 = (len(rhs_l.mesh.components[1]),)
    target_subshape1, target_subshape2 = Q.arg_index_shapes[:2]

    # Iterate over all points of the non-time mesh
    for non_t_idx in np.ndindex(non_time_mesh_shape):
        f_s = (slice(None), slice(None), *non_t_idx, Ellipsis)

        # Iterate over the second time argument and the second subshape of rhs_l
        for t2_idx, shape2_idx in product(np.ndindex(time_mesh_shape2),
                                          np.ndindex(target_subshape2)):

            rhs_s = (slice(None), *t2_idx,
                     *non_t_idx,
                     *[slice(None)] * len(target_subshape1), *shape2_idx)
            G_l.data[rhs_s] = solver(F_ret_ext.data[f_s], rhs_l.data[rhs_s])

    # Rebuild G from G_ret_ext and G_l
    return from_lesser_greater(G_l, G_l + G_ret_ext)
