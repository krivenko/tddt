"""
Hermiticity diagnostic for D-TRILEX quantities.

Runs the MM1 reference calculation step by step and measures the Keldysh
hermiticity violation at each stage:

    Stage 1 — Reference system:  G_imp, chi_imp (ch/sp), pi_imp (ch/sp), Lambda (ch/sp)
    Stage 2 — Bare dual:         g0.reg, w0[ch].reg, w0[sp].reg
    Stage 3 — Dressed dual:      pi_dual[ch].reg, pi_dual[sp].reg,
                                  w[ch].reg,      w[sp].reg
    Stage 4 — Self-energy:       sigma.reg

The hermiticity condition for a 2-point Keldysh GF G is (NESSi convention):
    G = herm_conj(G)   →   G^{ab}(t,t') = -[G^{ba}(t',t)]*

Violation is measured as:
    viol(G) = max_{a,b,t,t'} |G^{ab}(t,t') - herm_conj(G)^{ab}(t,t')|

For a clean quantity this should be ≲ 1e-12 (floating-point noise).

For Lambda (3-point vertex) we measure the weaker condition
    symm(Λ) = max |Λ^{abc}(t1,t2,t3) + conj(Λ^{bac}(t2,t1,t3))|
which checks that swapping the two fermionic time arguments and conjugating
gives minus the original (the expected crossing symmetry of the vertex).

Usage:
    python programs/debug_hermiticity.py [--t1 2.0] [--t_max 8.0] [--n_t 51]
"""

import argparse
import os
import sys

if __package__ is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

# MPI must be initialised before any TRIQS/realevol import that uses MPI types.
try:
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
except Exception:
    pass

import numpy as np
from triqs.gf import MeshReTime, MeshProduct

from tddt.keldysh import Branch, KeldyshGF, herm_conj
from programs.mickey_mouse import MickeyMouseModel
from programs.dual_quantities import DualQuantities, KeldyshGF_2_components

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Hermiticity diagnostic for D-TRILEX")
_parser.add_argument("--t1",   type=float, default=1.0,  help="System hopping (stress test)")
_parser.add_argument("--t_max",type=float, default=8.0,  help="Max real time")
_parser.add_argument("--n_t",  type=int,   default=51,   help="Number of time points")
_args, _ = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "-" * 72


def herm_viol(G: KeldyshGF, name: str) -> float:
    """
    Compute and print max|G - herm_conj(G)| over all Keldysh components.

    Returns the maximum violation as a float.
    """
    assert G.n_args == 2, f"{name}: expected 2-point GF, got n_args={G.n_args}"
    G_hc = herm_conj(G)
    mx = 0.0
    for b1 in (Branch.FORWARD, Branch.BACKWARD):
        for b2 in (Branch.FORWARD, Branch.BACKWARD):
            diff = G[b1, b2].data - G_hc[b1, b2].data
            mx = max(mx, float(np.max(np.abs(diff))))
    label = "FF/FB/BF/BB"
    print(f"  {name:<40s}  max|G - G†| = {mx:.3e}")
    return mx


def lambda_symm_viol(Lam: KeldyshGF, name: str) -> float:
    """
    For a 3-point vertex Λ(t1,t2,t3), measure the crossing symmetry violation:
        max |Λ^{abc}(t1,t2,t3) - conj(Λ^{bac}(t2,t1,t3))|

    The expected relation Λ(1,2,3) = conj(Λ(2,1,3)) comes from the fact that
    Λ = -⟨T c c† ρ⟩_conn (see Eq. 10), and swapping the two fermionic legs
    with conjugation should give a minus sign.
    """
    assert Lam.n_args == 3, f"{name}: expected 3-point GF, got n_args={Lam.n_args}"
    flip = lambda b: Branch(1 - b.value)
    mx = 0.0
    branches = (Branch.FORWARD, Branch.BACKWARD)
    for b1 in branches:
        for b2 in branches:
            for b3 in branches:
                L   = Lam[b1, b2, b3].data   # shape (n_t, n_t, n_t, 1, 1, 1)
                # Lam^{bac}(t2,t1,t3):  swap axes 0↔1 of the time indices
                L_swap = Lam[flip(b2), flip(b1), flip(b3)].data.swapaxes(0, 1)
                diff = L - np.conj(L_swap)
                mx = max(mx, float(np.max(np.abs(diff))))
    print(f"  {name:<40s}  max|Λ-conj(Λ̃)| = {mx:.3e}")
    assert mx < 1e-6, \
        f"{name}: Lambda Hermitian symmetry violated: max error = {mx:.3e}"
    return mx


def section(title: str):
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

t_max = _args.t_max
n_t   = _args.n_t
t1    = _args.t1

print(f"\nD-TRILEX hermiticity diagnostic")
print(f"  t1={t1},  t_max={t_max},  n_t={n_t},  dt={t_max/(n_t-1):.4f}")

t_mesh = MeshReTime(0, t_max, n_t)

U0, U, U1 = 2.0, 2.0, 2.0
t = 1.0

ref_params = {
    "U0": U0, "U": U, "mu": 0.5 * U, "t": t, "eps": 0.0,
    "T": 5.0, "bath_occ": 0.5, "n_particles": 2,
}
sys_params = {
    "U0": U0, "U": U1, "mu": 0.5 * U1, "t": t1, "eps": 0.0,
    "T": 5.0, "bath_occ": 0.5, "n_particles": 2,
}

params = {
    "verbosity": 0,
    "hbar": 1.0,
    "hamiltonian_interpol": "Trapezoid",
    "lanczos_min_matrix_size": 40,
}

# ---------------------------------------------------------------------------
# Stage 1: Reference system quantities
# ---------------------------------------------------------------------------

section("STAGE 1 — Reference system quantities")

Mickey_ref = MickeyMouseModel(ref_params, t_mesh)
Mickey_sys = MickeyMouseModel(sys_params, t_mesh)

print("\n[1a] After computeGandGimp:")
Mickey_ref.computeGandGimp(params)
herm_viol(Mickey_ref.g_imp, "G_imp (ref)")

print("\n[1b] After computeChiAndPi:")
Mickey_ref.computeChiAndPi(t_mesh, params)
for ch in ["ch", "sp"]:
    herm_viol(Mickey_ref.chi_imp[ch], f"chi_imp[{ch}]")
    herm_viol(Mickey_ref.pi_imp[ch],  f"pi_imp[{ch}]")

print("\n[1c] After computeLambda:")
Mickey_ref.computeLambda(t_mesh, params)
for ch in ["ch", "sp"]:
    lambda_symm_viol(Mickey_ref.Lambda[ch], f"Lambda[{ch}]")

# ---------------------------------------------------------------------------
# Stage 2: Bare dual propagators
# ---------------------------------------------------------------------------

section("STAGE 2 — Bare dual propagators (g0, w0)")

from programs.dual_quantities import KeldyshGF_2_components

delta_hyb = KeldyshGF_2_components(
    t_mesh, target_shape=(1, 1), is_reg=True,
    reg=Mickey_sys.hyb - Mickey_ref.hyb,
)
g_imp_ = KeldyshGF_2_components(
    t_mesh, target_shape=(1, 1), is_reg=True, reg=Mickey_ref.g_imp
)

dual_params = {
    "U_ch": 0.5 * U, "U_sp": -0.5 * U,
    "U_ch_sys": 0.5 * U1, "U_sp_sys": -0.5 * U1,
}

print("\n[2a] Building DualQuantities (g0, w0, pi_dual, w)...")
dual = DualQuantities(
    t_mesh, dual_params, delta_hyb, g_imp_,
    Mickey_ref.pi_imp, Mickey_ref.Lambda,
)

print("\n[2b] Bare dual GF g0:")
herm_viol(dual.g0.reg, "g0.reg")

print("\n[2c] Bare dual boson w0:")
for ch in ["ch", "sp"]:
    herm_viol(dual.w0[ch].reg, f"w0[{ch}].reg")

# ---------------------------------------------------------------------------
# Stage 3: Dressed bosonic quantities (pi_dual, w)
# ---------------------------------------------------------------------------

section("STAGE 3 — Dressed bosonic quantities (pi_dual, w)")

print("\n[3a] Dual bosonic self-energy pi_dual:")
for ch in ["ch", "sp"]:
    herm_viol(dual.pi[ch].reg, f"pi_dual[{ch}].reg")

print("\n[3b] Dressed dual boson w:")
for ch in ["ch", "sp"]:
    herm_viol(dual.w[ch].reg, f"w[{ch}].reg")

# ---------------------------------------------------------------------------
# Stage 4: Fermionic self-energy sigma
# ---------------------------------------------------------------------------

section("STAGE 4 — Dual fermionic self-energy Σ̃")

print("\n[4] Computing sigma (hartree_on=False)...")
dual.computeFermionSelfEnergy(Mickey_ref.Lambda, hartree_on=False)
herm_viol(dual.sigma.reg, "sigma.reg")

# Also check sigma diagonal (t1=t2) growth — the "exponential growth" observable
print("\n[4b] max|sigma(t,t)| vs time (diagonal, FB component):")
sigma_data = dual.sigma.reg[Branch.FORWARD, Branch.BACKWARD].data  # (n_t, n_t, 1, 1)
t_vals = np.linspace(0, t_max, n_t)
print(f"  {'t':>8s}  {'max|sigma(t,t)|':>18s}")
for i in range(0, n_t, max(1, n_t // 10)):
    val = abs(sigma_data[i, i, 0, 0])
    print(f"  {t_vals[i]:8.3f}  {val:18.6e}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

section("SUMMARY")

print(f"\n{'Quantity':<45s}  {'Violation':>12s}")
print("-" * 60)

violations = {}

violations["G_imp"]         = herm_viol(Mickey_ref.g_imp, "G_imp")
for ch in ["ch", "sp"]:
    violations[f"chi[{ch}]"]  = herm_viol(Mickey_ref.chi_imp[ch], f"chi_imp[{ch}]")
    violations[f"pi[{ch}]"]   = herm_viol(Mickey_ref.pi_imp[ch],  f"pi_imp[{ch}]")
violations["g0"]            = herm_viol(dual.g0.reg,        "g0.reg")
for ch in ["ch", "sp"]:
    violations[f"w0[{ch}]"]      = herm_viol(dual.w0[ch].reg,  f"w0[{ch}].reg")
    violations[f"pi_dual[{ch}]"] = herm_viol(dual.pi[ch].reg,  f"pi_dual[{ch}].reg")
    violations[f"w[{ch}]"]       = herm_viol(dual.w[ch].reg,   f"w[{ch}].reg")
violations["sigma"]         = herm_viol(dual.sigma.reg, "sigma.reg")

print(f"\n{'Quantity':<45s}  {'Violation':>12s}")
print("-" * 60)
for name, v in violations.items():
    flag = "  <<<  BUG" if v > 1e-8 else ("  (warn)" if v > 1e-12 else "")
    print(f"  {name:<43s}  {v:12.3e}{flag}")

print(f"\nDone. If any violation > 1e-8 is flagged, that is the first")
print(f"non-hermitian quantity and likely the root cause of sigma growth.")
