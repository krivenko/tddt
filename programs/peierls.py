"""
Peierls phase for the inter-cluster t_⊥ bond in the Mickey-Mouse² model.

The field is oriented along the t_⊥ bond, so the phase reduces to:

  φ_⊥(t) = exp(-i A0 sin(Ωt))

where A0 absorbs the bond length and polarisation projection (both gauge-trivial
constants that can be folded into the amplitude).

Hermiticity: φ_⊥*(t) appears on the conjugate hopping by construction.
"""

import numpy as np


def phase_tperp(t, A0, Omega):
    """
    Peierls phase for the inter-cluster t_⊥ bond.

    φ_⊥(t) = exp(-i A0 sin(Ωt))
    """
    return np.exp(-1j * A0 * np.sin(Omega * t))


# ---------------------------------------------------------------------------
# Sanity check and plot
# ---------------------------------------------------------------------------

def _sanity_check(A0=1.5, Omega=1.0):
    print("=== Peierls phase sanity checks ===")
    print(f"  phase_tperp(0) = {phase_tperp(0.0, A0, Omega):.6f}  (expected 1+0j)")
    t_test = 0.7
    ph = phase_tperp(t_test, A0, Omega)
    print(f"  |phase_⊥({t_test})|² = {abs(ph)**2:.10f}  (expected 1.0)")


def _plot(A0=1.5, Omega=1.0, output="tmp/peierls_test.png"):
    import os
    import matplotlib.pyplot as plt

    T_period = 2 * np.pi / Omega
    t_arr = np.linspace(0, 2 * T_period, 500)
    ph = phase_tperp(t_arr, A0, Omega)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_arr, ph.real, label=r"Re[$\phi_\perp$]")
    ax.plot(t_arr, ph.imag, label=r"Im[$\phi_\perp$]")
    ax.plot(t_arr, np.abs(ph), "k--", label=r"$|\phi_\perp|$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\phi_\perp(t)$")
    ax.set_title(rf"$A_0={A0}$, $\Omega={Omega}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"  Plot saved to {output}")


if __name__ == "__main__":
    _sanity_check()
    print()
    _plot()
