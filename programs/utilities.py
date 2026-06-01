import numpy as np

from tddt.keldysh import KeldyshGF
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# -----------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------

def _branch_key(*branches):
    """
    Build a compact string key from one or more Branch enum values.

    Examples: (FORWARD, BACKWARD) → 'FB',  (FORWARD, FORWARD, BACKWARD) → 'FFB'.
    Used as dataset names inside h5 groups so every Keldysh component has a
    human-readable label.
    """
    return ''.join(b.name[0] for b in branches)


def save_keldysh_gf_2pt(h5group, name, G: KeldyshGF):
    """
    Save all four Keldysh components of a 2-point Green's function G(t1,t2)
    into an h5py group.

    Creates a sub-group h5group[name] containing four complex128 datasets:
        FF  — G[FORWARD,  FORWARD ].data   shape (n_t, n_t, ...)
        FB  — G[FORWARD,  BACKWARD].data
        BF  — G[BACKWARD, FORWARD ].data
        BB  — G[BACKWARD, BACKWARD].data

    Parameters
    ----------
    h5group : h5py.Group  — parent group to write into.
    name    : str         — name of the sub-group to create.
    G       : KeldyshGF   — 2-point Keldysh Green's function.
    """
    from tddt.keldysh import Branch
    grp = h5group.require_group(name)
    for b1 in Branch:
        for b2 in Branch:
            grp.create_dataset(_branch_key(b1, b2), data=G[b1, b2].data)


def save_keldysh_gf_3pt(h5group, name, G: KeldyshGF):
    """
    Save all eight Keldysh components of a 3-point Green's function G(t1,t2,t3)
    into an h5py group.

    Creates a sub-group h5group[name] with eight complex128 datasets:
        FFF, FFB, FBF, FBB, BFF, BFB, BBF, BBB

    Parameters
    ----------
    h5group : h5py.Group  — parent group to write into.
    name    : str         — name of the sub-group to create.
    G       : KeldyshGF   — 3-point Keldysh Green's function (e.g. Lambda).
    """
    from tddt.keldysh import Branch
    grp = h5group.require_group(name)
    for b1 in Branch:
        for b2 in Branch:
            for b3 in Branch:
                grp.create_dataset(_branch_key(b1, b2, b3), data=G[b1, b2, b3].data)


def print_GF(filename, G, brf, brb, t_mesh, idx0=0):
    """
    Write a single Keldysh component of a Green's function to a text file.

    The output file has three columns:
        t    Re[G(t, t')]    Im[G(t, t')]
    where t' is the fixed second time argument selected by idx0 (the index
    into the second time axis of G.data).  Only one Keldysh component is
    written at a time, selected by (brf, brb):
        brf = Branch.FORWARD   — forward Keldysh branch for the first time
        brb = Branch.BACKWARD  — backward Keldysh branch for the second time

    Parameters
    ----------
    filename : str
        Path to the output file (created or overwritten).
    G : KeldyshGF
        The two-time Green's function to dump.
    brf : Branch
        Keldysh branch index for the first time argument.
    brb : Branch
        Keldysh branch index for the second time argument.
    t_mesh : MeshReTime
        The real-time mesh used to label the rows.
    idx0 : int, optional
        Index along the second time axis (fixes t').  Default is 0, which
        corresponds to t' = 0.
    """
    with open(filename, 'w') as fh:
        for t in t_mesh:
            try:
                val = G[brf, brb].data[idx0, t.index]
            except Exception:
                # If the data cannot be accessed for this time point, write zeros.
                val = 0.0j
            # G.data may carry extra target-shape axes (e.g. orbital indices).
            # Reduce to a single complex number: take the first element.
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    val = 0.0j
                else:
                    val = val.ravel()[0]
            fh.write(f"{t.value:.15e} {val.real:.15e} {val.imag:.15e}\n")
