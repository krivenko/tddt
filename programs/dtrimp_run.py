"""
D-TRILEX driver — single-shot, models 1 and 2.

Selects between the single-site (Mickey-Mouse 1, --model 1) and the
two-site (Mickey-Mouse², --model 2) D-TRILEX calculation via a single
command-line switch.  All physics parameters are read from a param file
whose path is given as the first positional argument (sys.argv[1]).

Usage
-----
    mpirun -n N python dtrimp_run.py param.in [--model {1,2}] [options]

Common parameters (param file or CLI)
--------------------------------------
    --model         {1,2}  which model to run  (default: 1)
    --U0            float  interaction before the quench
    --U             float  reference interaction after the quench
    --U1            float  system interaction after the quench
    --n_t           int    number of real-time grid points
    --t_max         float  maximum real time
    --eps_array     float+ bath site energies
    --t_ref_array   float+ reference hoppings (one per bath site)
    --t_sys_array   float+ system hoppings   (one per bath site)
    --T             float  temperature 1/β
    --hartree_on           include Hartree-Fock tadpole diagram

Model-2-only parameters
------------------------
    --t_perp        float  inter-site hopping t_⊥
    --V             float  inter-site Coulomb interaction V

Output parameters
-----------------
    --output_dir    str    directory for all output files  (default: programs/data)
    --output_name   str    base name (no extension)        (default: dtrimp_results)
    --simple_output        write plain-text .dat files instead of HDF5

Flat-bath parameters (sys only)
--------------------------------
    --flat_sys_bath        replace discrete sys bath sites with a flat-DOS bath;
                           G_sys becomes a dummy single-site ED result
    --D             float  half-bandwidth of the flat bath (required with --flat_sys_bath)
"""

import argparse
import shlex
import sys
import os
import time

import numpy as np

# -----------------------------------------------------------------------
# MPI setup
# -----------------------------------------------------------------------
_has_mpi = False
_comm    = None
_rank    = 0
_size    = 1

try:
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    _has_mpi = True
    _comm    = MPI.COMM_WORLD
    _rank    = _comm.Get_rank()
    _size    = _comm.Get_size()
except Exception:
    pass


def get_time():
    if _has_mpi:
        return MPI.Wtime()
    return time.perf_counter()


# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
from triqs.gf import MeshReTime, MeshProduct, Gf

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

if __package__ is None:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root  = os.path.dirname(_script_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import h5py
from programs.utilities import save_keldysh_gf_2pt, save_keldysh_gf_3pt, print_GF
from programs.dual_quantities import (
    DualQuantities, DualQuantities2,
    KeldyshGF_2_components, KeldyshGF_2x2_components,
)
from programs.mickey_mouse import MickeyMouseModel, MickeyMouseModel2
from tddt.keldysh import Branch, Singular2PKeldyshGF


# -----------------------------------------------------------------------
# Helper: build 2×2 Δ̃ and g_imp for model 2
# -----------------------------------------------------------------------
def _build_delta_gimp_2(t_mesh, target_shape, t_perp, g_imp, delta_hyb):
    """
    Construct the 2×2 Δ̃ and reference GF for the Mickey-Mouse² dual transform.

    Δ̃_∥ = delta_hyb  (single-site hybridisation difference)
    Δ̃_⊥ = t_⊥        (inter-site hopping, as a singular contact term)
    g_imp_matrix_∥ = g_imp,  g_imp_matrix_⊥ = 0
    """
    zero_fun  = KeldyshGF_2_components(t_mesh, target_shape, is_reg=True)
    g_imp_fun = KeldyshGF_2_components(t_mesh, target_shape, is_reg=True, reg=g_imp)

    t_perp_gf = Gf(mesh=t_mesh, target_shape=target_shape)
    for t in t_mesh:
        t_perp_gf[t] = t_perp
    t_perp_sing = Singular2PKeldyshGF.from_retime(t_perp_gf)

    delta_perp = KeldyshGF_2_components(t_mesh, target_shape, is_reg=False, sing=t_perp_sing)

    delta_tilde_matrix = KeldyshGF_2x2_components(par=delta_hyb, perp=delta_perp)
    g_imp_matrix       = KeldyshGF_2x2_components(par=g_imp_fun, perp=zero_fun)
    return delta_tilde_matrix, g_imp_matrix


# -----------------------------------------------------------------------
# Helper: plain-text output
# -----------------------------------------------------------------------
def _write_simple(out_dir, model, t_mesh, n_t,
                  G_ref, G_sys, G_dtrilex, G_cpt, Sigma_dtrilex):

    brf, brb = Branch.FORWARD, Branch.BACKWARD
    idx = 0

    def write_both_components(filename_base, G):
        """
        Write lesser (FB) and greater (BF) components.

        Produces:
            filename_base.dat
            filename_base_greater.dat
        """

        # Lesser component (FB)
        print_GF(
            os.path.join(out_dir, f"{filename_base}.dat"),
            G,
            brf,
            brb,
            t_mesh,
            idx0=idx
        )

        # Greater component (BF)
        print_GF(
            os.path.join(out_dir, f"{filename_base}_greater.dat"),
            G,
            brb,
            brf,
            t_mesh,
            idx0=idx
        )

    if model == 1:

        write_both_components("G_ref",         G_ref)
        write_both_components("G_sys",         G_sys)
        write_both_components("G_Dtrilex",     G_dtrilex)
        write_both_components("G_cpt",         G_cpt)
        write_both_components("Sigma_Dtrilex", Sigma_dtrilex)

    else:

        write_both_components("G_ref",                  G_ref)

        write_both_components("G_sys_par",             G_sys.par.reg)
        write_both_components("G_sys_perp",            G_sys.perp.reg)

        write_both_components("G_Dtrilex_par",         G_dtrilex.par.reg)
        write_both_components("G_Dtrilex_perp",        G_dtrilex.perp.reg)

        write_both_components("G_cpt_par",             G_cpt.par.reg)
        write_both_components("G_cpt_perp",            G_cpt.perp.reg)

        write_both_components("Sigma_Dtrilex_par",     Sigma_dtrilex.par.reg)
        write_both_components("Sigma_Dtrilex_perp",    Sigma_dtrilex.perp.reg)

    print(f"Done — plain-text files written to {out_dir}/")


# -----------------------------------------------------------------------
# Helper: HDF5 output
# -----------------------------------------------------------------------
def _write_h5(h5_path, model, t_max, n_t,
              Mickey_ref, dual_quantities,
              G_ref, G_sys, G_dtrilex, G_cpt):
    print(f"Writing results to {h5_path} ...")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("t_mesh", data=np.linspace(0, t_max, n_t))

        ref = f.require_group("ref")
        save_keldysh_gf_2pt(ref, "G_ref", G_ref)
        chi_grp = ref.require_group("chi")
        lam_grp = ref.require_group("Lambda")
        for ch in ["ch", "sp"]:
            save_keldysh_gf_2pt(chi_grp, ch, Mickey_ref.chi_imp[ch])
            save_keldysh_gf_3pt(lam_grp, ch, Mickey_ref.Lambda[ch])

        dual    = f.require_group("dual")
        sys_grp = f.require_group("sys")

        if model == 1:
            save_keldysh_gf_2pt(dual, "g_dual",    dual_quantities.g0.reg)
            save_keldysh_gf_2pt(dual, "sigma_dual", dual_quantities.sigma.reg)
            w_grp  = dual.require_group("w_dual")
            pi_grp = dual.require_group("pi_dual")
            for ch in ["ch", "sp"]:
                save_keldysh_gf_2pt(w_grp,  ch, dual_quantities.w0[ch].reg)
                save_keldysh_gf_2pt(pi_grp, ch, dual_quantities.pi[ch].reg)

            save_keldysh_gf_2pt(sys_grp, "G_sys",     G_sys)
            save_keldysh_gf_2pt(sys_grp, "G_dtrilex", G_dtrilex)
            save_keldysh_gf_2pt(sys_grp, "G_cpt",     G_cpt)

        else:
            def _save_par_perp(grp, name, G, boson=False):
                sub = grp.require_group(name)
                if not boson:
                    save_keldysh_gf_2pt(sub, "parallel",     G.par.reg)
                    save_keldysh_gf_2pt(sub, "perpendicular", G.perp.reg)
                else:
                    sub_par  = sub.require_group("parallel")
                    sub_perp = sub.require_group("perpendicular")
                    for ch in ["ch", "sp"]:
                        save_keldysh_gf_2pt(sub_par,  ch, G[ch].par.reg)
                        save_keldysh_gf_2pt(sub_perp, ch, G[ch].perp.reg)

            _save_par_perp(dual, "sigma_dual",  dual_quantities.sigma)
            _save_par_perp(dual, "w_dual",      dual_quantities.w,  boson=True)
            _save_par_perp(dual, "pi_dual",     dual_quantities.pi, boson=True)

            _save_par_perp(sys_grp, "G_sys",     G_sys)
            _save_par_perp(sys_grp, "G_dtrilex", G_dtrilex)
            _save_par_perp(sys_grp, "G_cpt",     G_cpt)

    print(f"Done — {h5_path}")


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="D-TRILEX single-shot driver (models 1 and 2)",
)

_parser.add_argument("--model",        type=int,   choices=[1, 2], default=1)

_parser.add_argument("--U0",           type=float)
_parser.add_argument("--U",            type=float)
_parser.add_argument("--U1",           type=float)
_parser.add_argument("--n_t",          type=int,   help="Number of time points")
_parser.add_argument("--t_max",        type=float, help="Maximum real time")
_parser.add_argument("--eps_array",    type=float, nargs="+", help="Bath site energies")
_parser.add_argument("--t_ref_array",  type=float, nargs="+", help="Reference hoppings")
_parser.add_argument("--t_sys_array",  type=float, nargs="+", help="System hoppings")
_parser.add_argument("--T",            type=float, help="Temperature = 1/β")
_parser.add_argument("--hartree_on",   action="store_true", help="Include HF diagram")

_parser.add_argument("--t_perp",       type=float, default=0.0, help="Inter-site hopping t_⊥ (model 2)")
_parser.add_argument("--V",            type=float, default=0.0, help="Inter-site Coulomb V (model 2)")

_parser.add_argument("--field_on",     action="store_true", default=False,
                     help="Enable Peierls phase on t_⊥ (model 2 only, AC laser field).")
_parser.add_argument("--A0",    type=float, default=0.0,
                     help="Vector potential amplitude. Phase: exp(-i A0 sin(Ω t)).")
_parser.add_argument("--Omega", type=float, default=1.0,
                     help="Laser frequency Ω.")

_parser.add_argument("--flat_sys_bath", action="store_true", default=False,
                     help="Use flat-DOS bath for sys instead of discrete ED sites.")
_parser.add_argument("--D",            type=float, default=None,
                     help="Half-bandwidth of the flat bath (required with --flat_sys_bath).")

_parser.add_argument("--output_dir",   type=str,   default="programs/data",
                     help="Directory for all output files")
_parser.add_argument("--output_name",  type=str,   default="dtrimp_results",
                     help="Base name (no extension) for the HDF5 output file")
_parser.add_argument("--simple_output", action="store_true",
                     help="Write plain-text .dat files instead of HDF5")

# -----------------------------------------------------------------------
# MPI-safe argument parsing: rank 0 reads param file, broadcasts to all
# -----------------------------------------------------------------------
if _rank == 0:
    with open(sys.argv[1]) as f:
        lines = []
        for line in f:
            line = line.split("#", 1)[0]  # remove comments
            line = line.strip()
            if line:
                lines.append(line)

    _arg_list = shlex.split(" ".join(lines))
    _args = _parser.parse_args(_arg_list)
else:
    _args = None

if _has_mpi:
    _args = _comm.bcast(_args, root=0)

# -----------------------------------------------------------------------
# Unpack args
# -----------------------------------------------------------------------
model       = _args.model
t_max       = _args.t_max
n_t         = _args.n_t
U0          = _args.U0
U           = _args.U
U1          = _args.U1
eps_array   = _args.eps_array
t_ref_array = _args.t_ref_array
t_sys_array = _args.t_sys_array
T           = _args.T
t_perp      = _args.t_perp
V           = _args.V
field_on      = _args.field_on
A0            = _args.A0
Omega         = _args.Omega
flat_sys_bath = _args.flat_sys_bath
D_flat        = _args.D
output_dir  = _args.output_dir
output_name = _args.output_name

t_mesh   = MeshReTime(0, t_max, n_t)
tt_mesh  = MeshProduct(t_mesh, t_mesh)
ttt_mesh = MeshProduct(t_mesh, t_mesh, t_mesh)

ref_params = {
    "U0":        U0,
    "U":         U,
    "mu":        0.5 * U,
    "eps_array": eps_array,
    "t_array":   t_ref_array,
    "T":         T,
}
if flat_sys_bath:
    if D_flat is None:
        raise ValueError("--D is required when --flat_sys_bath is set.")
    sys_params = {
        "U0":        U0,
        "U":         U1,
        "mu":        0.5 * U1,
        # eps_array[0] = flat band center e0; t_sys_array[0] = t_flat.
        # n_bath is overridden to 0 inside MickeyMouseModel.__init__.
        "eps_array": eps_array,
        "t_array":   t_sys_array,
        "T":         T,
        "flat_bath": True,
        "D":         D_flat,
        "field_on":  False,
        "A0":        0.0,
        "Omega":     1.0,
    }
else:
    sys_params = {
        "U0":        U0,
        "U":         U1,
        "mu":        0.5 * U1,
        "eps_array": eps_array,
        "t_array":   t_sys_array,
        "T":         T,
    }

if field_on and len(eps_array) != 1:
    raise ValueError(
        "--field_on requires exactly 1 bath site; "
        f"got {len(eps_array)} (--eps_array has {len(eps_array)} entries)."
    )

# Peierls phase on t_⊥ applies only to the system (model 2).
# Ref is always static — bath Peierls is gauge-trivial everywhere.
# Flat bath has no Peierls phase (field_on already set to False above).
if not flat_sys_bath:
    sys_params["field_on"] = field_on
    sys_params["A0"]       = A0
    sys_params["Omega"]    = Omega

if model == 2:
    sys_params["t_perp"] = t_perp
    sys_params["V"]      = V

dual_params = {
    "U_ch":      0.5 * U,
    "U_sp":     -0.5 * U,
    "U_ch_sys":  0.5 * U1,
    "U_sp_sys": -0.5 * U1,
}
if model == 2:
    dual_params["V"] = V

solver_params = {
    "verbosity":               2,
    "hbar":                    1.0,
    "hamiltonian_interpol":    "Trapezoid",
    "lanczos_min_matrix_size": 40,
}

if _rank == 0:
    print(f"Running with MPI: {_has_mpi}, size: {_size}, rank: {_rank}")
    print(f"Model: {model}")

# -----------------------------------------------------------------------
# Step 1: Reference impurity model (MPI-collective)
# -----------------------------------------------------------------------
Mickey_ref = MickeyMouseModel(ref_params, t_mesh)

if model == 1:
    Mickey_sys = MickeyMouseModel(sys_params, t_mesh)
else:
    Mickey_sys = MickeyMouseModel2(sys_params, t_mesh)

t_start = get_time()

if _rank == 0:
    print("Computing G_ref")
Mickey_ref.computeGandGimp(solver_params)

if _rank == 0:
    print(f"Done G_ref  [Time: {get_time() - t_start:.3f} s]")
    print("Computing Chi and Pi")
Mickey_ref.computeChiAndPi(t_mesh, solver_params)

if _rank == 0:
    print(f"Done Chi and Pi  [Total elapsed: {get_time() - t_start:.3f} s]")
    print("Computing Lambda")
Mickey_ref.computeLambda(t_mesh, solver_params)

t_impurity_done = get_time() - t_start
if _rank == 0:
    print(f"Done Lambda  [Impurity quantities total time: {t_impurity_done:.3f} s]")
    print("Computing target system GF...")

t_sys_start = get_time()
Mickey_sys.computeGandGimp(solver_params)

# Only rank 0 does the rest
if _has_mpi and _size > 1:
    _comm.Barrier()
    if _rank != 0:
        sys.exit(0)

# -----------------------------------------------------------------------
# Step 2: Dual quantities
# -----------------------------------------------------------------------
target_shape = (1, 1)

delta_hyb = KeldyshGF_2_components(
    t_mesh, target_shape=target_shape, is_reg=True,
    reg=Mickey_sys.hyb - Mickey_ref.hyb,
)
g_imp_ = KeldyshGF_2_components(
    t_mesh, target_shape=target_shape, is_reg=True,
    reg=Mickey_ref.g_imp,
)

if model == 2:
    delta_hyb_, g_imp_ = _build_delta_gimp_2(
        t_mesh, target_shape, t_perp, Mickey_ref.g_imp, delta_hyb
    )
else:
    delta_hyb_ = delta_hyb

if _rank == 0:
    print("Setting up dual quantities...")
t_dual_start = get_time()

if model == 1:
    dual_quantities = DualQuantities(
        t_mesh, dual_params, delta_hyb_, g_imp_,
        Mickey_ref.pi_imp, Mickey_ref.Lambda,
    )
else:
    dual_quantities = DualQuantities2(
        t_mesh, dual_params, delta_hyb_, g_imp_,
        Mickey_ref.pi_imp, Mickey_ref.Lambda,
    )

t_dual_setup = get_time() - t_dual_start
if _rank == 0:
    print(f"Dual quantities setup done  [Time: {t_dual_setup:.3f} s]")

# -----------------------------------------------------------------------
# Step 3: Dual self-energy
# -----------------------------------------------------------------------
if _rank == 0:
    print("Computing dual self-energy...")
t_se_start = get_time()

plots_dir = os.path.join(output_dir, "plots")
dual_quantities.computeFermionSelfEnergy(
    Mickey_ref.Lambda, hartree_on=_args.hartree_on, plots_dir=plots_dir
)

t_se = get_time() - t_se_start
if _rank == 0:
    print(f"Dual self-energy done  [Time: {t_se:.3f} s]")

# -----------------------------------------------------------------------
# Step 4: Physical Green's functions
# -----------------------------------------------------------------------
G_ref = Mickey_ref.g_imp
G_sys = Mickey_sys.g_imp

if model == 1:
    G_dtrilex     = Mickey_ref.computeFermionPropagatorFromDual(delta_hyb_.reg, dual_quantities.sigma.reg)
    G_cpt         = Mickey_ref.computeFermionPropagatorFromDual(delta_hyb_.reg)
    Sigma_dtrilex = dual_quantities.sigma.reg
else:
    G_dtrilex     = Mickey_ref.computeFermionPropagatorFromDual2(delta_hyb_, dual_quantities, g_imp_)
    G_cpt         = Mickey_ref.computeFermionPropagatorFromDual2(delta_hyb_, dual_quantities, g_imp_, is_cpt=True)
    Sigma_dtrilex = dual_quantities.sigma

t_sys   = get_time() - t_sys_start
t_total = get_time() - t_start
if _rank == 0:
    print(f"Target system GF done  [Time: {t_sys:.3f} s]")
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Impurity quantities (G, Chi, Pi, Lambda): {t_impurity_done:.3f} s")
    print(f"Dual quantities setup:                    {t_dual_setup:.3f} s")
    print(f"Dual self-energy:                         {t_se:.3f} s")
    print(f"Target system GF:                         {t_sys:.3f} s")
    print(f"TOTAL TIME:                               {t_total:.3f} s")
    print(f"===================\n")

# -----------------------------------------------------------------------
# Step 5: Output
# -----------------------------------------------------------------------
if _rank == 0:
    os.makedirs(output_dir, exist_ok=True)

    if _args.simple_output:
        _write_simple(
            output_dir, model, t_mesh, n_t,
            G_ref, G_sys, G_dtrilex, G_cpt, Sigma_dtrilex,
        )
    else:
        h5_path = os.path.join(output_dir, output_name + ".h5")
        _write_h5(
            h5_path, model, t_max, n_t,
            Mickey_ref, dual_quantities,
            G_ref, G_sys, G_dtrilex, G_cpt,
        )
