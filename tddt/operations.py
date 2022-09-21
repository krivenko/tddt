#
# Operations with correlators on Keldysh contour
#

from copy import deepcopy
import numpy as np
from mpmath import mp

from triqs.gf import MeshReFreq, GfReTime

from .keldysh import Branch, KeldyshGF

def make_legendre_transform_matrix(N: int):
    r"""
    Make a Legendre polynomial basis to equispaced real points transformation
    matrix

    P_{il} = \sqrt{2l+1} P_l(x_i),  x_i = -1 + 2*i / (N - 1)
    """
    assert N <= 101, "Larger values of N are untested with the chosen mp.dps"
    mp.dps = 60

    x = lambda i: -1 + mp.mpf(2*i) / mp.mpf(N - 1)

    P = mp.matrix(N, N)
    for l in range(N):
        for i in range(N):
            P[i, l] = mp.sqrt(2*l+1) * mp.legendre(l, x(i))

    return P


def invert_keldysh_correlator_2t(chi: KeldyshGF):
    """Invert a 2-point correlator defined on a Keldysh contour"""

    assert chi.target_shape == ()

    t_max = chi.time_mesh[0].t_max
    n_t = len(chi.time_mesh[0])

    mp.dps = 60
    P = make_legendre_transform_matrix(n_t)
    invP = P**-1

    P /= t_max
    invP *= t_max

    FW = Branch.FORWARD
    BW = Branch.BACKWARD

    print(np.linalg.norm(chi[FW, FW].data - np.conj(chi[BW, BW].data)))
    print(np.linalg.norm(chi[FW, BW].data - np.conj(chi[BW, FW].data)))

    #print(np.diag(chi[FW, BW].data) - np.diag(chi[BW, FW].data))
    #print(np.triu(chi[BW, FW].data, k=1) + np.tril(chi[FW, BW].data, k=0) - chi[BW, BW].data)
    #print(np.triu(chi[BW, FW].data, k=0) + np.tril(chi[FW, BW].data, k=-1) - chi[BW, BW].data)

    print(chi[FW, FW].data - np.transpose(chi[FW, FW].data))
    print(chi[BW, BW].data - np.transpose(chi[BW, BW].data))
    print(chi[BW, FW].data - np.transpose(chi[FW, BW].data))


    # Matrix to be inverted
    mat = mp.matrix(2*n_t, 2*n_t)

    mat[:n_t, :n_t] = invP @ mp.matrix(chi[Branch.FORWARD, Branch.FORWARD].data) @ invP.T
    mat[:n_t, n_t:] = -invP @ mp.matrix(chi[Branch.FORWARD, Branch.BACKWARD].data) @ invP.T
    mat[n_t:, :n_t] = invP @ mp.matrix(chi[Branch.BACKWARD, Branch.FORWARD].data) @ invP.T
    mat[n_t:, n_t:] = -invP @ mp.matrix(chi[Branch.BACKWARD, Branch.BACKWARD].data) @ invP.T

    # FIXME
    print(mat)

    mat /= t_max**2

    # FIXME
    def cond(m):
        S = mp.svd(m, compute_uv=False)
        print(S)
        return S[-1] / S[0]


    #print(cond(mat[:n_t, :n_t]))
    #print(cond(mat[:n_t, n_t:]))
    #print(cond(mat[n_t:, :n_t]))
    #print(cond(mat[n_t:, n_t:]))
    print(cond(mat))

    #inv_chi_mat = mat**-1
    #
    #print(mp.mnorm(inv_chi_mat @ mat - mp.eye(2*n_t), p=mp.inf))

    #inv_chi = deepcopy(chi)
    #inv_chi[Branch.FORWARD, Branch.FORWARD].data[:] = P @ inv_chi_mat[:n_t, :n_t] @ P.T
    #inv_chi[Branch.FORWARD, Branch.BACKWARD].data[:] = P @ inv_chi_mat[:n_t, n_t:] @ P.T
    #inv_chi[Branch.BACKWARD, Branch.FORWARD].data[:] = P @ inv_chi_mat[n_t:, :n_t] @ P.T
    #inv_chi[Branch.BACKWARD, Branch.BACKWARD].data[:] = P @ inv_chi_mat[n_t:, n_t:] @ P.T
    #
    #return inv_chi
