#
# Auxiliary routines for 2-point Green's functions of real time
#

import numpy as np

from triqs.gf import Gf, MeshReTime, MeshProduct

from .util import subscripts
from .integration import GregoryIntegrator


def _is_2t_gf(g: Gf) -> bool:
    """Is g a 2-point real time GF?"""
    return len(g.mesh.components) >= 2 and \
        isinstance(g.mesh.components[0], MeshReTime) and \
        isinstance(g.mesh.components[1], MeshReTime)


def _extract_nli_nri(g: Gf, n_left_indices, assert_msg):
    """Return numbers of left and right indices in a real time GF."""
    if n_left_indices is None:
        assert len(g.target_shape) % 2 == 0, assert_msg
        nli = len(g.target_shape) // 2
    else:
        nli = n_left_indices

    nri = len(g.target_shape) - nli

    return nli, nri


def conj(g: Gf, *, n_left_indices=None) -> Gf:
    r"""
    Given a 2-point real time Green's function G_{a, b}(t, t'), returns its
    Hermitian conjugate [G_{b, a}(t', t)]^*. The conjugation is performed
    independently for each point of the non-time components of G's mesh.

    g: Input Green's function.
    n_left_indices: Number of axes in G's target shape corresponding to the
                    multi-index 'a'. By default, a half of all axes.
    """
    assert _is_2t_gf(g)

    mesh = MeshProduct(g.mesh.components[1],
                       g.mesh.components[0],
                       *g.mesh.components[2:])

    nli, nri = _extract_nli_nri(
        g,
        n_left_indices,
        "n_left_indices must be provided when the target shape of the GF "
        "has an odd number of dimensions"
    )

    ts = g.target_shape[nli:] + g.target_shape[:nli]

    g_conj = Gf(mesh=mesh, target_shape=ts)
    axes_from = [0, 1, *range(-1, - nri - 1, -1)]
    axes_to = [1, 0, *range(-1 - nli, - nri - nli - 1, -1)]
    g_conj.data[:] = np.conj(np.moveaxis(g.data, axes_from, axes_to))

    return g_conj


def _make_nontime_mesh_and_subs(a_mesh, b_mesh):
    nt_mesh_a = a_mesh.components[2:]
    nt_mesh_b = b_mesh.components[2:]

    nts = subscripts['nontime']
    if nt_mesh_a == nt_mesh_b:
        # If the non-time components of the meshes agree, then we
        # use the same non-time mesh for the result
        ss = nts[:len(nt_mesh_a)]
        subs_a_nt = ss
        subs_b_nt = ss
        subs_res_nt = ss
        mesh_comps_res = nt_mesh_a
    else:
        # Otherwise the result is defined on a direct product of the meshes.
        ss = nts[:len(nt_mesh_a)]
        subs_a_nt = ss
        subs_res_nt = ss
        ss = nts[len(nt_mesh_a):len(nt_mesh_a) + len(nt_mesh_b)]
        subs_b_nt = ss
        subs_res_nt += ss
        mesh_comps_res = nt_mesh_a + nt_mesh_b

    return subs_a_nt, subs_b_nt, subs_res_nt, mesh_comps_res


def _make_target_shape_subs(a_nli, a_nri, b_nli, b_nri):
    tgs = subscripts['target']
    subs_a_tg = tgs[:a_nli + a_nri]
    subs_b_tg = tgs[a_nli:(a_nli + b_nli + b_nri)]
    subs_res_tg = tgs[:a_nli]
    subs_res_tg += tgs[(a_nli + a_nri):(a_nli + b_nli + b_nri)]
    return subs_a_tg, subs_b_tg, subs_res_tg


def conv_ret_ret(a_ret: Gf,
                 b_ret: Gf,
                 a_ret_n_left_indices=None,
                 gregory_order=5) -> Gf:
    r"""
    Compute a real-time convolution of two extended retarded functions,

    F_{a, b}(t, t') =
        \sum_c \int_{t'}^t d\bar t A^r_{a, c}(t, \bar t) B^r_{c, b}(\bar t, t').

    a_ret: The extended retarded function A^r(t, t').
    b_ret: The extended retarded function B^r(t, t').
    a_ret_n_left_indices: Number of axes in A^r's target shape corresponding to
                          the multi-index 'a'. By default, a half of all axes.
    gregory_order: Order of the Gregory quadrature rule used to do the
                   convolution.
    """
    assert _is_2t_gf(a_ret)
    assert _is_2t_gf(b_ret)

    # Handle the time components of the meshes
    assert a_ret.mesh.components[0] == a_ret.mesh.components[1], \
           "The two components of a_ret's time mesh must be the same"
    assert b_ret.mesh.components[0] == b_ret.mesh.components[1], \
           "The two components of b_ret's time mesh must be the same"
    assert a_ret.mesh.components[1] == b_ret.mesh.components[0], \
           "Incompatible time meshes of a_ret and b_ret"

    t_mesh = a_ret.mesh.components[0]

    # Handle the non-time components of the meshes
    subs_a_ret_nt, subs_b_ret_nt, subs_res_nt, nt_mesh_comps_res = \
        _make_nontime_mesh_and_subs(a_ret.mesh, b_ret.mesh)

    mesh = MeshProduct(t_mesh, t_mesh, *nt_mesh_comps_res)

    # Handle the targets
    a_ret_nli, a_ret_nri = _extract_nli_nri(
        a_ret,
        a_ret_n_left_indices,
        "a_ret_n_left_indices must be provided when the target shape of A^r "
        "has an odd number of dimensions"
    )

    b_ret_nli = a_ret_nri
    b_ret_nri = len(b_ret.target_shape) - b_ret_nli

    assert a_ret.target_shape[a_ret_nli:] == b_ret.target_shape[:b_ret_nli], \
        "Incompatible target shapes of a_ret and b_ret"

    subs_a_ret_tg, subs_b_ret_tg, subs_res_tg = \
        _make_target_shape_subs(a_ret_nli, a_ret_nri, b_ret_nli, b_ret_nri)

    target_shape_res = \
        a_ret.target_shape[:a_ret_nli] + b_ret.target_shape[b_ret_nli:]

    # Generate einsum() subscripts
    ts = subscripts['time']
    subs_a_ret_t = ts[1]
    subs_b_ret_t = ts[1] + ts[0]
    subs_res_t = ts[0]

    subs_a_ret = subs_a_ret_t + subs_a_ret_nt + subs_a_ret_tg
    subs_b_ret = subs_b_ret_t + subs_b_ret_nt + subs_b_ret_tg
    subs_res = subs_res_t + subs_res_nt + subs_res_tg

    # Perform summation (Eq. (110) of the NESSi paper).
    res = Gf(mesh=mesh, target_shape=target_shape_res)
    integrator = GregoryIntegrator(gregory_order)

    subs_I = ts[0] + ts[1]
    k_slice = slice(integrator.order + 1)
    delta_I = t_mesh.delta * integrator.I

    subs = f"{subs_a_ret},{subs_I},{subs_b_ret}->{subs_res}"
    # Eq. (110), line 3
    for n in range(integrator.order + 1):
        np.einsum(subs,
                  a_ret.data[n, k_slice, ...],
                  delta_I[:(n + 1), n, :],
                  b_ret.data[k_slice, :(n + 1), ...],
                  out=res.data[n, :(n + 1), ...],
                  optimize="optimal")

    w = integrator.weights(t_mesh)

    subs_a_ret = ts[0] + subs_a_ret_nt + subs_a_ret_tg
    subs_b_ret = ts[0] + subs_b_ret_nt + subs_b_ret_tg
    subs_res = subs_res_nt + subs_res_tg
    subs_w = ts[0]
    subs = f"{subs_a_ret},{subs_w},{subs_b_ret}->{subs_res}"

    for n in range(integrator.order + 1, len(t_mesh)):
        # Eq. (110), line 1
        for m in range(n - integrator.order):
            np.einsum(subs,
                      a_ret.data[n, m:(n + 1), ...],
                      w[n - m, :(n - m + 1)],
                      b_ret.data[m:(n + 1), m, ...],
                      out=res.data[n, m, ...],
                      optimize="optimal")
        # Eq. (110), line 2
        for m in range(n - integrator.order, n + 1):
            np.einsum(subs,
                      a_ret.data[n, n:(n - integrator.order - 1):-1, ...],
                      w[n - m, k_slice],
                      b_ret.data[n:(n - integrator.order - 1):-1, m, ...],
                      out=res.data[n, m, ...],
                      optimize="optimal")

    return res


def conv_ret_lg(a_ret: Gf,
                b_lg: Gf,
                a_ret_n_left_indices=None,
                gregory_order=5) -> Gf:
    r"""
    Compute a real-time convolution of an extended retarded function and a
    lesser or greater function,

    F_{a, b}(t, t') =
        \sum_c \int_0^t d\bar t A^r_{a, c}(t, \bar t) B^{<>}_{c, b}(\bar t, t').

    a_ret: The extended retarded function A^r(t, t').
    b_l: The lesser or greater function B^{<>}(t, t').
    a_ret_n_left_indices: Number of axes in A^r's target shape corresponding to
                          the multi-index 'a'. By default, a half of all axes.
    gregory_order: Order of the Gregory quadrature rule used to do the
                   convolution.
    """
    assert _is_2t_gf(a_ret)
    assert _is_2t_gf(b_lg)

    # Handle the time components of the meshes
    assert a_ret.mesh.components[1] == b_lg.mesh.components[0], \
           "Incompatible time meshes of a_ret and b_lg"
    assert len(a_ret.mesh.components[0]) >= len(a_ret.mesh.components[1]), \
           "First component of a_ret's mesh cannot be shorter than the second"

    ts = subscripts['time']
    subs_a_ret_t = ts[0] + ts[2]
    subs_b_lg_t = ts[2] + ts[1]
    subs_res_t = ts[0] + ts[1]
    subs_w = subs_a_ret_t
    t_mesh_comps_res = [a_ret.mesh.components[0], b_lg.mesh.components[1]]

    # Handle the non-time components of the meshes
    subs_a_ret_nt, subs_b_lg_nt, subs_res_nt, nt_mesh_comps_res = \
        _make_nontime_mesh_and_subs(a_ret.mesh, b_lg.mesh)

    mesh = MeshProduct(*t_mesh_comps_res, *nt_mesh_comps_res)

    # Handle the targets
    a_ret_nli, a_ret_nri = _extract_nli_nri(
        a_ret,
        a_ret_n_left_indices,
        "a_ret_n_left_indices must be provided when the target shape of A^r "
        "has an odd number of dimensions"
    )

    b_lg_nli = a_ret_nri
    b_lg_nri = len(b_lg.target_shape) - b_lg_nli

    assert a_ret.target_shape[a_ret_nli:] == b_lg.target_shape[:b_lg_nli], \
        "Incompatible target shapes of a_ret and b_lg"

    subs_a_ret_tg, subs_b_lg_tg, subs_res_tg = \
        _make_target_shape_subs(a_ret_nli, a_ret_nri, b_lg_nli, b_lg_nri)

    target_shape_res = \
        a_ret.target_shape[:a_ret_nli] + b_lg.target_shape[b_lg_nli:]

    # Generate einsum() subscripts
    subs_a_ret = subs_a_ret_t + subs_a_ret_nt + subs_a_ret_tg
    subs_b_lg = subs_b_lg_t + subs_b_lg_nt + subs_b_lg_tg
    subs_res = subs_res_t + subs_res_nt + subs_res_tg

    subs = f"{subs_a_ret},{subs_w},{subs_b_lg}->{subs_res}"

    # Perform summation
    res = Gf(mesh=mesh, target_shape=target_shape_res)

    # Eqs. (117)-(118) of the NESSi paper.
    w = GregoryIntegrator(gregory_order).weights(a_ret.mesh.components[1])
    res.data[:] = np.einsum(subs,
                            a_ret.data,
                            w[:len(t_mesh_comps_res[0]), :],
                            b_lg.data, optimize="optimal")

    return res


def conv_lg_adv(a_lg: Gf,
                b_adv: Gf,
                a_lg_n_left_indices=None,
                gregory_order=5) -> Gf:
    r"""
    Compute a real-time convolution of a lesser or greater function and an
    extended advanced function,

    F_{a, b}(t, t') =
      \sum_c \int_0^{t'} d\bar t A^{<>}_{a, c}(t, \bar t) B^a{c, b}(\bar t, t').

    a_lg: The lesser or greater function A^{<>}(t, t').
    b_l: The extended advanced function B^a(t, t').
    a_lg_n_left_indices: Number of axes in A^{<>}'s target shape corresponding
                         to the multi-index 'a'. By default, a half of all axes.
    gregory_order: Order of the Gregory quadrature rule used to do the
                   convolution.
    """
    assert _is_2t_gf(a_lg)
    assert _is_2t_gf(b_adv)

    # Handle the time components of the meshes
    assert a_lg.mesh.components[1] == b_adv.mesh.components[0], \
           "Incompatible time meshes of a_lg and b_adv"
    assert len(b_adv.mesh.components[1]) >= len(b_adv.mesh.components[0]), \
           "Second component of b_adv's mesh cannot be shorter than the first"

    ts = subscripts['time']
    subs_a_lg_t = ts[0] + ts[2]
    subs_b_adv_t = ts[2] + ts[1]
    subs_res_t = ts[0] + ts[1]
    subs_w = ts[1] + ts[2]
    t_mesh_comps_res = [a_lg.mesh.components[0], b_adv.mesh.components[1]]

    # Handle the non-time components of the meshes
    subs_a_lg_nt, subs_b_adv_nt, subs_res_nt, nt_mesh_comps_res = \
        _make_nontime_mesh_and_subs(a_lg.mesh, b_adv.mesh)

    mesh = MeshProduct(*t_mesh_comps_res, *nt_mesh_comps_res)

    # Handle the targets
    a_lg_nli, a_lg_nri = _extract_nli_nri(
        a_lg,
        a_lg_n_left_indices,
        "a_lg_n_left_indices must be provided when the target shape of A^{<>} "
        "has an odd number of dimensions"
    )

    b_adv_nli = a_lg_nri
    b_adv_nri = len(b_adv.target_shape) - b_adv_nli

    assert a_lg.target_shape[a_lg_nli:] == b_adv.target_shape[:b_adv_nli], \
        "Incompatible target shapes of a_lg and b_adv"

    subs_a_lg_tg, subs_b_adv_tg, subs_res_tg = \
        _make_target_shape_subs(a_lg_nli, a_lg_nri, b_adv_nli, b_adv_nri)

    target_shape_res = \
        a_lg.target_shape[:a_lg_nli] + b_adv.target_shape[b_adv_nli:]

    # Generate einsum() subscripts
    subs_a_lg = subs_a_lg_t + subs_a_lg_nt + subs_a_lg_tg
    subs_b_adv = subs_b_adv_t + subs_b_adv_nt + subs_b_adv_tg
    subs_res = subs_res_t + subs_res_nt + subs_res_tg

    subs = f"{subs_a_lg},{subs_w},{subs_b_adv}->{subs_res}"

    # Perform summation
    res = Gf(mesh=mesh, target_shape=target_shape_res)

    # Eqs. (119)-(120) of the NESSi paper.
    w = GregoryIntegrator(gregory_order).weights(a_lg.mesh.components[1])
    res.data[:] = np.einsum(subs,
                            a_lg.data,
                            w[:len(t_mesh_comps_res[1]), :],
                            b_adv.data,
                            optimize="optimal")

    return res
