# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# Copyright (C) 2021-2026, I. Krivenko, V. Harkov, V. Valmispild
#
# ##############################################################################

"""I/O helpers for cluster2x2_run.py output."""

from tddt.keldysh import Branch

_BRANCH_KEYS_2PT = {
    (Branch.FORWARD,  Branch.FORWARD):  'FF',
    (Branch.FORWARD,  Branch.BACKWARD): 'FB',
    (Branch.BACKWARD, Branch.FORWARD):  'BF',
    (Branch.BACKWARD, Branch.BACKWARD): 'BB',
}


def write_keldysh_gf_file(filename, g, spc_point=None, target_indices=()):
    """
    Write (FW,FW) and (FW,BW) components of a 2-point KeldyshGF to a text file.

    Each row: Re(FW,FW)  Im(FW,FW)  Re(FW,BW)  Im(FW,BW)
    Rows correspond to the second time argument t; the first is fixed at t=0.
    """
    FW, BW = Branch
    t_mesh = g.time_mesh.components[1]
    t0 = next(iter(t_mesh))
    with open(filename, 'w') as fh:
        fh.write("# Re (FW, FW) Im (FW, FW) Re (FW, BW) Im (FW, BW)\n")
        for t in t_mesh:
            mesh_point = (t0, t) if (spc_point is None) else (t0, t, spc_point)
            col_data = (
                g[FW, FW][*mesh_point][*target_indices].real,
                g[FW, FW][*mesh_point][*target_indices].imag,
                g[FW, BW][*mesh_point][*target_indices].real,
                g[FW, BW][*mesh_point][*target_indices].imag,
            )
            fh.write("{} {} {} {}\n".format(*col_data))


def save_keldysh_gf_2pt_h5(h5group, name, g):
    """
    Save all 4 Keldysh components of a 2-point KeldyshGF to an HDF5 group.

    Creates h5group[name]/FF, FB, BF, BB as complex128 datasets containing
    g[b1, b2].data (shape: (n_t, n_t [, n_k], *target_shape)).
    """
    grp = h5group.require_group(name)
    for (b1, b2), key in _BRANCH_KEYS_2PT.items():
        grp.create_dataset(key, data=g[b1, b2].data)
