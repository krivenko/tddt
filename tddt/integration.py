#
# Numerical integration tools
#

import numpy as np
from scipy.special import binom

from triqs.gf import MeshReTime


def simpsons_weights(mesh: MeshReTime):
    """
    Make a list of Simpson’s rule weights for integration on a real time mesh.
    """
    w = np.ones(len(mesh))
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    w *= mesh.delta / 3
    return w


def gregory_coefficients(n_max):
    r"""
    Compute a list of Gregory coefficients G_1, G_2, ..., G_{n_max}
    using a recurrence formula

    \frac{G_1}{n} - \frac{G_2}{n-1} + \frac{G_3}{n-2} - ... +
        + (-1)^{n-1}\frac{G_n}{1} = \frac{1}{n+1}.
    """
    assert n_max >= 1

    G = [0.5]
    for n in range(2, n_max + 1):
        s = sum((-1) ** (i - 1) / (n + 1 - i) * G[i - 1] for i in range(1, n))
        G_n = (-1) ** (n - 1) * (1 / (n + 1) - s)
        G.append(G_n)

    return np.array(G)


class GregoryIntegrator:
    """
    This class implements the Gregory quadrature rule as described in
    M. Schüler et al, Comp. Phys. Comm. 257, 107484 (2020).
    """

    def __init__(self, order):
        self.order = order

        self.I = self.poly_integral_coeffs(order)  # noqa: E741
        self.s = self.I[0, :, :]
        self.B = self.boundary_correction_weights(order)

    @classmethod
    def poly_integral_coeffs(cls, k):
        r"""
        Computes the (k+1)x(k+1)x(k+1) tensor of polynomial integration
        coefficients I^{(k)}_{m,n,l}.
        """
        assert k >= 0
        k_range = range(k + 1)

        # Vandermonde matrix j^a, j=0,...,k, a=0,...,k
        M = np.vander(k_range, increasing=True)

        # Coefficients of k-th polynomial interpolation P^{(k)}_{a,l}
        P_k = np.linalg.inv(M)

        # Matrix of the integration operator
        m, n, a = np.meshgrid(k_range, k_range, k_range, indexing='ij')
        int_op = (n ** (a + 1) - m ** (a + 1)) / (a + 1)

        return np.tensordot(int_op, P_k, axes=1)

    @classmethod
    def boundary_correction_weights(cls, k):
        r"""
        Computes (k+1) Gregory boundary correction weights B^{k}_j.
        """
        assert k >= 0

        G = gregory_coefficients(k + 1)

        def f(r, i):
            return G[r - 1] * (-1) ** (r - i) * binom(r - 1, i)
        return np.array(
            [sum(f(r, i) for r in range(2, k + 2)) for i in range(k + 1)]
        )

    def weights(self, mesh: MeshReTime):
        r"""
        Returns the weights matrix w^{(k)}_{n,j} for integration on a given
        real time mesh.
        """
        n = len(mesh)
        stencil_size = self.order + 1
        assert n >= stencil_size

        # The Trapezoid rule contribution
        w = np.tri(n, k=0)
        w[:, 0] = 0.5
        np.fill_diagonal(w, 0.5)

        # Gregory boundary corrections
        if self.order > 0:
            # Left boundary correction
            w[:, :stencil_size] += self.B

            # Right boundary correction
            i, j = np.indices(w.shape)
            for d in range(stencil_size):
                w[j == i - d] += self.B[d]

        # Patch in the starting block
        w[:stencil_size, :stencil_size] = self.s

        w *= mesh.delta
        return w

    def weights_conv(self, mesh: MeshReTime):
        r"""
        Returns the weights for a complete contour integral.
        """
        n = len(mesh)
        stencil_size = self.order + 1
        assert n >= stencil_size

        if n == stencil_size:
            return mesh.delta * self.s[-1, :]

        # The Trapezoid rule contribution
        w = np.ones(n)
        w[0] = w[-1] = 0.5

        # Gregory boundary corrections
        if self.order > 0:
            w[:stencil_size] += self.B
            w[-stencil_size:] += np.flip(self.B)

        w *= mesh.delta
        print(w)
        return w
