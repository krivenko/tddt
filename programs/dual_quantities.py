"""
Dual quantities for the D-TRILEX method on the Keldysh contour.

This module implements the dual transformation described in the notes.
The key idea is to map the interacting system onto a set of dual
fermions and bosons whose bare propagators encode the
difference between the reference (Mickey-Mouse) and the target system.

DualQuantities  — for the single-site (Mickey-Mouse 1) model.
DualQuantities2 — extends DualQuantities to the two-site (Mickey-Mouse²) model.

The D-TRILEX self-consistency loop is:
    1. Compute fermionic self-energy Σ̃ from Λ, G̃, W̃  (Eq. 21/24)
    2. Update dual GF G̃ via the Dyson equation              (Eq. 26)
    3. Compute bosonic self-energy Π̃ from Λ, G̃             (Eq. 22/25)
    4. Update dual boson W̃ via the Dyson equation           (Eqs. 28-29)
    5. Repeat until convergence.
"""

from itertools import product
import numpy as np

from triqs.gf import Gf, MeshReTime
from triqs.gf import MeshProduct

from tddt.keldysh import KeldyshGF, conv, Branch, herm_conj, Singular2PKeldyshGF

from tddt.vie2 import solve_vie2

from new_keldysh_classes import (
    KeldyshGF_2_components, KeldyshGF_2x2_components,
    _sing_times_reg, _reg_times_sing, _lambda_times_sing, _sing_times_lambda,
)


def plot_herm_viol(G, name: str, save_dir: str = ".") -> None:
    """Save a 4×2 imshow of |G - G†| and |G - G†|/|G| per Keldysh block to save_dir."""
    if isinstance(G, KeldyshGF_2x2_components):
        plot_herm_viol(G.par.reg,  f"{name}[par]",  save_dir)
        plot_herm_viol(G.perp.reg, f"{name}[perp]", save_dir)
        return
    if isinstance(G, KeldyshGF_2_components):
        plot_herm_viol(G.reg, name, save_dir)
        return
    import matplotlib.pyplot as plt
    import os

    G_hc = herm_conj(G)
    blocks = [
        (Branch.FORWARD,  Branch.FORWARD,  "FF"),
        (Branch.FORWARD,  Branch.BACKWARD, "FB"),
        (Branch.BACKWARD, Branch.FORWARD,  "BF"),
        (Branch.BACKWARD, Branch.BACKWARD, "BB"),
    ]

    times = np.array([float(t) for t in G.mesh.components[0]])
    t0, t1_max = times[0], times[-1]
    extent = [t0, t1_max, t0, t1_max]

    fig, axes = plt.subplots(4, 2, figsize=(11, 16))
    fig.suptitle(f"Hermiticity violation  —  {name}")

    for row, (b1, b2, label) in enumerate(blocks):
        data = G[b1, b2].data
        diff = np.abs(data - G_hc[b1, b2].data)
        scale = np.max(np.abs(data))

        flat_abs = diff.reshape(diff.shape[0], diff.shape[1], -1).max(axis=-1)
        flat_rel = (diff / (scale + 1e-30)).reshape(diff.shape[0], diff.shape[1], -1).max(axis=-1)

        ax_abs, ax_rel = axes[row, 0], axes[row, 1]

        im0 = ax_abs.imshow(flat_abs, origin="lower", aspect="auto", cmap="hot_r", extent=extent)
        ax_abs.set_title(f"{label}  |G - G†|")
        ax_abs.set_xlabel("t2"); ax_abs.set_ylabel("t1")
        fig.colorbar(im0, ax=ax_abs)

        im1 = ax_rel.imshow(flat_rel, origin="lower", aspect="auto", cmap="hot_r", extent=extent)
        ax_rel.set_title(f"{label}  |G - G†| / max|G|")
        ax_rel.set_xlabel("t2"); ax_rel.set_ylabel("t1")
        fig.colorbar(im1, ax=ax_rel)

    fig.tight_layout()
    safe_name = name.replace("/", "_").replace(" ", "_")
    path = os.path.join(save_dir, f"herm_viol_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  plot_herm_viol  {name}  →  {path}")


def herm_viol(G, name: str) -> float:
    """Print and return max|G - G†|; accepts KeldyshGF, KeldyshGF_2_components, or KeldyshGF_2x2_components."""
    if isinstance(G, KeldyshGF_2x2_components):
        v_par  = herm_viol(G.par.reg,  f"{name}[par]")
        v_perp = herm_viol(G.perp.reg, f"{name}[perp]")
        return max(v_par, v_perp)
    if isinstance(G, KeldyshGF_2_components):
        return herm_viol(G.reg, name)
    G_hc = herm_conj(G)
    mx_abs = 0.0
    scale = 0.0
    for b1 in (Branch.FORWARD, Branch.BACKWARD):
        for b2 in (Branch.FORWARD, Branch.BACKWARD):
            data = G[b1, b2].data
            mx_abs = max(mx_abs, float(np.max(np.abs(data - G_hc[b1, b2].data))))
            scale = max(scale, float(np.max(np.abs(data))))
    mx_rel = mx_abs / (scale + 1e-30)
    print(f"  herm_viol  {name:<35s}  max|G - G†| = {mx_abs:.3e}  rel = {mx_rel:.3e}")
    return mx_abs


class DualQuantities:
    """
    All dual propagators and self-energies for the Mickey-Mouse 1 model.

    After construction the object holds:
      self.g    — dual GF G̃  (Eq. 12), updated each iteration
      self.g0   — initial (bare) dual GF G̃⁰, kept fixed
      self.w0   — bare dual boson W̃⁰ (Eq. 17)
      self.w    — dual boson W̃, updated each iteration
      self.pi   — bosonic dual self-energy Π̃ (Eq. 22)
      self.sigma— fermionic dual self-energy Σ̃ (Eq. 21); None until computed

    Each propagator is stored as a KeldyshGF_2_components, i.e. split into
    a singular (one-time) part and a regular (two-time) part (see Eq. 15).
    The dict keys 'ch' and 'sp' label the charge and spin channels.
    """

    def __init__(self, time_mesh, dual_params, delta_tilde, g_imp, pi_imp, Lambda, *, init_g=None, init_w=None):
        """
        Initialise all bare dual propagators and run the first bosonic update.

        Parameters
        ----------
        time_mesh : MeshReTime
            Real-time mesh shared by all propagators.
        dual_params : dict
            Interaction parameters for both reference and system:
              'U_ch'     — charge interaction of the reference  (+U/2)
              'U_sp'     — spin interaction of the reference    (-U/2)
              'U_ch_sys' — charge interaction of the benchmark system
              'U_sp_sys' — spin interaction of the benchmark system
        delta_tilde : KeldyshGF_2_components
            Hybridisation difference Δ̃ = Δ_sys - Δ_ref  (Eq. 11).
        g_imp : KeldyshGF_2_components
            Impurity GF of the reference system g_imp (Eq. 2).
        pi_imp : dict of KeldyshGF
            Local polarisation Π_ς of the reference system (Eq. 13),
            keyed by channel ('ch', 'sp').
        Lambda : dict of KeldyshGF
            Three-point vertex Λ^ς of the reference system (Eq. 10),
            keyed by channel.
        init_g : KeldyshGF_2_components, optional
            Warm-start value for the dual GF G̃.  If None, G̃⁰ is used.
        init_w : dict of KeldyshGF_2_components, optional
            Warm-start values for the dual bosons W̃.  If None, W̃ is
            computed from scratch via the bosonic Dyson equation.
        """
        self.t_mesh = time_mesh
        self.tt_mesh = MeshProduct(time_mesh, time_mesh)

        self.target_shape = (1, 1)

        self.channels = ["ch", "sp"]
        # Reference and system interactions per channel.
        self.U = {"ch": dual_params["U_ch"], "sp": dual_params["U_sp"]}
        self.U_sys = {"ch": dual_params["U_ch_sys"], "sp": dual_params["U_sp_sys"]}
        # Singular part of the bare dual W̃^δ = U_sys - U/2  (Eq. 17).
        self.U_tilde = {
            channel: self.U_sys[channel] - 0.5 * self.U[channel]
            for channel in self.channels
        }

        self.delta_tilde = delta_tilde
        self.sigma = None  # filled by computeFermionSelfEnergy()

        # Compute bare dual GF G̃⁰ (Eq. 12); warm-start old times from init_g.
        self.g0 = self.setGdual(g_imp)
        herm_viol(self.g0, "g0")
        self.g = self.g0
        if init_g is not None:
            _fill_old_times_2_components(self.g, init_g)
        herm_viol(self.g, "g (init)")

        # Singular (contact) interactions U and U_sys as Singular2PKeldyshGF objects.
        self.w_sing = self.setW_bare(self.U)
        self.w_sys_sing = self.setW_bare(self.U_sys)
        self.w_tilde_sing = self.setW_bare(self.U_tilde)

        # Bosonic self-energy Π̃ using the current G̃.
        self.pi = {}
        self.computeBosonSelfEnergy(Lambda)

        # Bare dual boson W̃⁰ (Eq. 17); warm-start old times from init_w.
        self.w0 = self.setW0(pi_imp)
        herm_viol(self.w0["ch"], "w0[ch]")
        herm_viol(self.w0["sp"], "w0[sp]")
        self.w = {}
        self.computeBosonPropagators()
        if init_w is not None:
            for ch in self.channels:
                _fill_old_times_2_components(self.w[ch], init_w[ch])

    # ------------------------------------------------------------------
    # Core Dyson-equation solver
    # ------------------------------------------------------------------

    def solveVie2(self, A: KeldyshGF_2_components, B: KeldyshGF_2_components):
        """
        Solve the Dyson-type Volterra integral equation of the second kind:

            (1 - A) * X = B

        which appears in both the fermionic Dyson equation (Eq. 26) and the
        bosonic Dyson equation (Eqs. 28-29).

        The equation is solved only for the regular part; the singular part of
        X is inherited directly from B.  Before calling the numerical solver the
        right-hand side is hermitian-symmetrised to suppress round-off errors.

        Parameters
        ----------
        A : KeldyshGF_2_components
            Kernel of the integral equation.  Must be purely regular (is_reg=True).
        B : KeldyshGF_2_components
            Right-hand side / initial (bare) propagator.

        Returns
        -------
        KeldyshGF_2_components  — solution X with the same singular part as B.
        """
        if not A.is_reg:
            raise ValueError("A must be regular")
        # Symmetrise A.reg: reg@reg convolution of two hermitian GFs is not
        # hermitian in general (AB ≠ BA), so we enforce it before use.
        # A.reg appears in both the RHS (via _reg_times_sing) and the VIE kernel.
        A_reg = A.reg #0.5 * (A.reg + herm_conj(A.reg))
        A_reg_herm = herm_conj(A_reg)
        FQ = 0.5 * (_reg_times_sing(A_reg, B.sing) + _sing_times_reg(B.sing, A_reg_herm))
        herm_viol(FQ, "FQ")

        # Subtract the singular contribution to get the regular RHS.
        # Use exact right-multiplication instead of conv() to avoid quadrature error.
        Q = B.reg - FQ
        herm_viol(Q, "Q")
        # Symmetrise Q to suppress any residual floating-point asymmetry.
        Q = 0.5 * (Q + herm_conj(Q))
        result_reg = solve_vie2(A_reg, Q)
        #result_reg = 0.5 * (result_reg + herm_conj(result_reg))
        return KeldyshGF_2_components(
            self.t_mesh,
            self.target_shape,
            B.is_reg,
            sing=B.sing,
            reg=result_reg,
        )

    # ------------------------------------------------------------------
    # Initialisation of bare propagators
    # ------------------------------------------------------------------

    def setW_bare(self, U):
        """
        Wrap the scalar interaction U as a time-independent Singular2PKeldyshGF.

        This represents U as a contact (δ-function in time) propagator, which
        is the singular part W^δ = U of the bosonic propagator (Eq. 16/56).

        Parameters
        ----------
        U : dict  — interaction values keyed by channel ('ch', 'sp').

        Returns
        -------
        dict of Singular2PKeldyshGF, one per channel.
        """
        w0 = {}
        for channel in self.channels:
            # Build a trivial time-dependent Gf with constant value U[channel].
            tmp = Gf(mesh=self.t_mesh, target_shape=self.target_shape)
            for t in self.t_mesh:
                tmp[t] = U[channel]
            w0[channel] = Singular2PKeldyshGF.from_retime(tmp)
        return w0

    def setW0(self, pi_imp):
        """
        Compute the bare dual bosonic propagator W̃⁰ for each channel.

        From Eq. (17):
            W̃^δ_ς  = U^ς_sys - U^ς/2        (singular / contact part)
            W̃^reg_ς = W^reg_ς                 (regular part)

        where W^reg is the regular part of the reference bosonic propagator W_ς,
        obtained by solving (1 - U_sys * Π) * W^reg = U_sys * Π * U_sys  (Eq. 16).

        Parameters
        ----------
        U     : dict  — reference interactions per channel.
        U_sys : dict  — system interactions per channel.
        pi_imp: dict  — local polarisation Π_ς of the reference (Eq. 13).

        Returns
        -------
        dict of KeldyshGF_2_components, one per channel.
        """
        w0 = {}
        for channel in self.channels:
            w0[channel] = KeldyshGF_2_components(
                self.t_mesh, self.target_shape, is_reg=False
            )
            # Singular part: W̃^δ = U_sys - U/2  (Eq. 17)
            w0[channel].sing = self.w_tilde_sing[channel]

            # Regular part: solve (1 - U_sys * Π) * W^reg = U_sys * Π * U_sys  (Eq. 16)
            W = self.w_sys_sing[channel]
            W_pi = _sing_times_reg(W, pi_imp[channel])   # W @ pi, exact
            Q = _reg_times_sing(W_pi, W)                 # (W @ pi) @ W, exact
            herm_viol(Q, f'Q_W {channel}')
            Q = 0.5 * (Q + herm_conj(Q))  # enforce hermiticity
            w0[channel].reg = solve_vie2(-W_pi, Q)
        return w0

    def setGdual(self, g_imp):
        """
        Compute the bare dual GF G̃ from the reference impurity GF.

        Solves Eq. (12):
            (1 - Δ̃ * g_imp) * G̃ = Δ̃

        Parameters
        ----------
        g_imp : KeldyshGF_2_components
            Impurity GF of the reference system.

        Returns
        -------
        KeldyshGF_2_components  — bare dual GF G̃.
        """
        print("Updating bare fermionic propagator...")
        return self.solveVie2(-self.delta_tilde @ g_imp, self.delta_tilde)

    # ------------------------------------------------------------------
    # Intermediate convolution building blocks
    # The helpers below implement the intermediate quantities F1, F2, N
    # from footnote 1 of the notes (Eqs. 18-20), used to assemble Σ̃ and Π̃.
    # ------------------------------------------------------------------

    def compute_f1(self, Lambda, g_in):
        """
        F1(1,2,3) = ∫d4 Λ(1,4,3) G̃(4,2)   — first intermediate (Eq. 18).
        Contracts the first time argument of G̃ with the second argument of Λ.
        """
        return conv(Lambda, g_in, [(1, 0)], free_args=([0, 2], [1]))
    
    def compute_f1_herm(self, Lambda, g_in):
        """
        F1(4,1,3) = ∫d4 G̃(2,4) Λ(4,1,3)    — first intermediate (Eq. 18).
        Contracts the first time argument of G̃ with the second argument of Λ.
        """
        return conv(g_in, Lambda, [(1, 0)], free_args=([0], [1, 2]))

    def compute_f1_HF(self, Lambda, g_in):
        """
        Hartree-Fock variant of F1: contracts both time arguments of G̃ with Λ,
        yielding the local density-like quantity N (Eq. 20).
        """
        return conv(Lambda, g_in, [(0, 1), (1, 0)])

    def compute_f1_GW(self, Lambda, w_in):
        LW = conv(Lambda, w_in, [(2, 0)], free_args=([0, 1], [2]))
        return conv(LW, Lambda, [(2, 2)], free_args=([0, 2], [3, 1]))

    def compute_f2(self, Lambda, w_in):
        """
        F2(1,2,3) = ∫d4 Λ(1,2,4) W̃(3,4)   — second intermediate (Eq. 19).
        Contracts the third argument of Λ with the second argument of W̃.
        """
        if isinstance(w_in, Singular2PKeldyshGF):
            # W̃ = f(t)·δ_C  →  F2(t1,t2,t3) = Λ(t1,t2,t3)·f(t3), exact.
            return _lambda_times_sing(Lambda, w_in)
        return conv(Lambda, w_in, [(2, 1)], free_args=([0, 1], [2]))
    
    def compute_f2_herm(self, Lambda, w_in):
        """
        F2(1,2,3) = ∫d4 Λ(1,2,4) W̃(3,4)   — second intermediate (Eq. 19).
        Contracts the third argument of Λ with the second argument of W̃.
        """
        if isinstance(w_in, Singular2PKeldyshGF):
            # W̃ = f(t)·δ_C  →  F2(t1,t2,t3) = Λ(t1,t2,t3)·f(t3), exact.
            return _sing_times_lambda(w_in, Lambda)
        return conv(Lambda, w_in, [(2, 0)], free_args=([0, 1], [2]))

    def compute_f2_HF(self, w, n):
        """
        Hartree-Fock variant of F2: contracts W̃ with the local density N.
        """
        return conv(w, n, [(1, 0)])

    def compute_pi(self, f1, f2):
        """
        Dual bosonic self-energy kernel: Π̃ = -i ∫ F1 F1  (Eq. 22).
        Contracts the two F1 objects over their shared time indices.
        """
        return -1j * conv(f1, f2, [(0, 1), (1, 0)])

    def compute_sigma(self, f1, f2):
        """
        GW-like contribution to the dual self-energy: Σ̃_GW = i ∫ F1 F2  (Eq. 21).
        """
        return 1j * conv(f1, f2, [(2, 2), (1, 0)])

    def compute_sigma_GW(self, LWL, g_in):
        return conv(LWL, g_in, [(2, 0), (3, 1)], free_args=([0, 1],[]))
    
    def compute_sigma_HF(self, Lambda, f2):
        """
        Tadpole (Hartree-Fock) contribution to the dual self-energy (Eq. 21).
        """
        return -1j * conv(Lambda, f2, [(2, 0)])

    # ------------------------------------------------------------------
    # Generic second-order diagram assembler
    # ------------------------------------------------------------------
    def convolution2ndOrderGW(
        self,
        Lambda: KeldyshGF,
        g: KeldyshGF_2_components,
        w: KeldyshGF_2_components,
        compute_f1,
        compute,
    ):
        assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
        assert g.reg.n_args == 2, "g must be a 2-point GF"

        conv_result = KeldyshGF_2_components(
            self.t_mesh, self.target_shape, is_reg=True
        )

        # Select which components of g and w are non-zero.
        components_g = [g.reg] if g.is_reg else [g.sing, g.reg]
        components_w = [w.reg] if w.is_reg else [w.sing, w.reg]

        # Sum all non-zero combinations of singular and regular parts.
        for g1 in components_g:
            for w1 in components_w:
                LWL = compute_f1(Lambda, w1)
                assert LWL.n_args == 4, "Lambda must be a 3-point vertex"
                conv_result.reg += compute(LWL, g1)
        return conv_result

    def convolution2ndOrder(
        self,
        Lambda: KeldyshGF,
        g: KeldyshGF_2_components,
        w: KeldyshGF_2_components,
        compute_f1,
        compute_f2,
        compute,
    ):
        """
        Evaluate a general second-order diagram by summing over all singular/regular
        combinations of the input propagators.

        Any second-order diagram has the schematic form:
            Result = ∫ Λ · g · Λ · w   (or Λ · g · Λ · g for the polarisation)

        Both g and w are split into singular + regular parts (Eq. 15), so the
        full convolution requires four terms: (sing,sing), (sing,reg),
        (reg,sing), (reg,reg).  This method iterates over all active combinations
        and accumulates them into the regular part of the result.

        Parameters
        ----------
        Lambda       : KeldyshGF             — three-point vertex Λ^ς.
        g            : KeldyshGF_2_components — dual fermionic propagator G̃.
        w            : KeldyshGF_2_components — dual bosonic propagator W̃ (or G̃ for Π̃).
        compute_f1   : callable              — builds the F1 intermediate from (Λ, g).
        compute_f2   : callable              — builds the F2 intermediate from (Λ, w).
        compute      : callable              — combines F1 and F2 into the diagram value.

        Returns
        -------
        KeldyshGF_2_components — diagram result (purely regular, is_reg=True).
        """
        assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
        assert g.reg.n_args == 2, "g must be a 2-point GF"

        conv_result = KeldyshGF_2_components(
            self.t_mesh, self.target_shape, is_reg=True
        )

        # Select which components of g and w are non-zero.
        components_g = [g.reg] if g.is_reg else [g.sing, g.reg]
        components_w = [w.reg] if w.is_reg else [w.sing, w.reg]

        # Sum all non-zero combinations of singular and regular parts.
        for g1 in components_g:
            for w1 in components_w:
                f1 = compute_f1(Lambda, g1)
                f2 = compute_f2(Lambda, w1)
                sigma_contrib = compute(f1, f2)
                conv_result.reg += sigma_contrib
        return conv_result

    def convolution2ndOrderHF(
        self,
        Lambda: KeldyshGF,
        g: KeldyshGF_2_components,
        w: KeldyshGF_2_components,
        compute_f1,
        compute_f2,
        compute,
    ):
        """
        Evaluate the Hartree-Fock (tadpole) second-order diagram.

        The tadpole diagram differs from the GW diagram in that the bosonic
        line W̃ is contracted with the local density N = ∫ Λ G̃ (Eq. 20)
        rather than with F2.  The structure is:
            Σ̃_HF ~ -i Λ · (W̃ · N)

        Same singular/regular decomposition logic as convolution2ndOrder.

        Parameters
        ----------
        Lambda, g, w : same as convolution2ndOrder.
        compute_f1   : callable — builds the local density N from (Λ, g).
        compute_f2   : callable — contracts W̃ with N.
        compute      : callable — assembles the final Σ̃_HF from (Λ, W̃·N).

        Returns
        -------
        KeldyshGF_2_components — Hartree-Fock self-energy contribution.
        """
        assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
        assert g.reg.n_args == 2, "g must be a 2-point GF"

        conv_result = KeldyshGF_2_components(
            self.t_mesh, self.target_shape, is_reg=True
        )

        components_g = [g.reg] if g.is_reg else [g.sing, g.reg]
        components_w = [w.reg] if w.is_reg else [w.sing, w.reg]

        for g1 in components_g:
            for w1 in components_w:
                f1 = compute_f1(Lambda, g1)       # N = ∫ Λ G̃
                f2 = compute_f2(w1, f1)            # W̃ · N
                conv_result.reg += compute(Lambda, f2)
        return conv_result

    # ------------------------------------------------------------------
    # Physical self-energy computations
    # ------------------------------------------------------------------

    def computeBosonSelfEnergy(self, Lambda):
        """
        Compute the dual bosonic self-energy Π̃_ς for each channel.

        Implements Eq. (22):
            Π̃_ς(1,2) = -i ∑_{σσ'} ∫d(34) F1^{σσ'}_ς(3,4,1) F1^{σ'σ}_ς(4,3,2)

        The factor of 2 accounts for the sum over the two spin components
        in the paramagnetic case (σ = ↑↓, both equal by assumption).

        Result is stored in self.pi, keyed by channel.

        Parameters
        ----------
        Lambda : dict of KeldyshGF — three-point vertices per channel.
        """
        for channel in self.channels:
            # Factor 2: sum over spin (↑↑ + ↓↓, both equal in paramagnetic case).
            self.pi[channel] = self.convolution2ndOrder(
                Lambda[channel],
                self.g,
                self.g,         # bosonic self-energy uses G̃ twice (Eq. 22)
                self.compute_f1,
                self.compute_f1,
                self.compute_pi,
            )
            herm_viol(self.pi[channel], f"pi[{channel}]")
            # Symmetrise: Gregory quadrature gives O(dt^6) violations; suppress them.
            #self.pi[channel].reg = 0.5 * (
            #    self.pi[channel].reg + herm_conj(self.pi[channel].reg)
            #)

    def computeFermionSelfEnergy(self, Lambda, hartree_on=True, plots_dir="."):
        """
        Compute the total dual fermionic self-energy Σ̃ = Σ̃_GW + Σ̃_HF.

        The GW-like diagram is the main D-TRILEX contribution (Eq. 21, first line).
        The Hartree-Fock (tadpole) diagram is the second line of Eq. (21) and
        captures the static mean-field correction.

        Result is stored in self.sigma.

        Parameters
        ----------
        Lambda    : dict of KeldyshGF — three-point vertices per channel.
        hartree_on: bool, optional    — whether to include the HF tadpole diagram.
                                        Default is True.
        plots_dir : str, optional     — directory for the herm_viol_sigma plot.
                                        Default is the current working directory.
        """
        Sigma_GW = self.computeGWDiagram(Lambda)
        if hartree_on:
            Sigma_HF = self.computeHFDiagram(Lambda)
            self.sigma = Sigma_GW + Sigma_HF
        else:
            self.sigma = Sigma_GW
        # 3-point×3-point Gregory quadrature in compute_sigma introduces O(dt^p)
        # hermiticity violation. Implement using Langreth decomposition to preserve
        # hermiticity by computing sigma_ret and sigma_< separately.
        #self.sigma.reg = 0.5 * (self.sigma.reg + herm_conj(self.sigma.reg))
        herm_viol(self.sigma, "sigma")
        import os as _os
        _os.makedirs(plots_dir, exist_ok=True)
        plot_herm_viol(self.sigma, "sigma", save_dir=plots_dir)

    def computeGWDiagram(self, Lambda):
        """
        Compute the GW-like diagram contribution to Σ̃ (first line of Eq. 21):

            Σ̃_GW = i ∑_{σ',ς} c_ς ∫d(34) F1^{σσ'}_ς(1,3,4) F2^{σ'σ}_ς(3,2,4)

        The spin-channel coefficients are c_ch = 1, c_sp = 3, which arise from
        the spin summation and channel decomposition of the Coulomb interaction.

        Parameters
        ----------
        Lambda : dict of KeldyshGF — three-point vertices per channel.

        Returns
        -------
        KeldyshGF_2_components — GW contribution to the dual self-energy.
        """
        # Coefficients from the spin summation: charge channel × 1, spin channel × 3.
        coeff = {"ch": 1, "sp": 3}
        Sigma_GW = KeldyshGF_2_components(self.t_mesh, target_shape=self.target_shape, is_reg=True)
        for channel in self.channels:
            Sigma_GW += coeff[channel] * self.convolution2ndOrder(
                Lambda[channel],
                self.g,
                self.w[channel],
                self.compute_f1,
                self.compute_f2,
                self.compute_sigma,
            )
        return Sigma_GW
    
    def computeGWDiagram_2(self, Lambda):
        """
        Compute the GW-like diagram contribution to Σ̃ (first line of Eq. 21):

            Σ̃_GW = i ∑_{σ',ς} c_ς ∫d(34) F1^{σσ'}_ς(1,3,4) F2^{σ'σ}_ς(3,2,4)

        The spin-channel coefficients are c_ch = 1, c_sp = 3, which arise from
        the spin summation and channel decomposition of the Coulomb interaction.

        Parameters
        ----------
        Lambda : dict of KeldyshGF — three-point vertices per channel.

        Returns
        -------
        KeldyshGF_2_components — GW contribution to the dual self-energy.
        """
        # Coefficients from the spin summation: charge channel × 1, spin channel × 3.
        coeff = {"ch": 1, "sp": 3}
        Sigma_GW = KeldyshGF_2_components(self.t_mesh, target_shape=self.target_shape, is_reg=True)
        for channel in self.channels:
            Sigma_GW += coeff[channel] * self.convolution2ndOrderGW(
                Lambda[channel],
                self.g,
                self.w[channel],
                self.compute_f1_GW,
                self.compute_sigma_GW,
            )
        return Sigma_GW

    def computeHFDiagram(self, Lambda):
        """
        Compute the Hartree-Fock (tadpole) contribution to Σ̃ (second line of Eq. 21):

            Σ̃_HF = -i ∑_{σ',ς} 2 c_ς ∫d(34) Λ^{σσ}_ς(1,2,3) W̃⁰_ς(3,4) N^{σ'}(4)

        Here W̃⁰ is the bare dual boson (not the fully dressed W̃) because the
        tadpole diagram only requires the static part of the interaction.
        The extra factor of 2 comes from the spin sum over σ'.

        Parameters
        ----------
        Lambda : dict of KeldyshGF — three-point vertices per channel.

        Returns
        -------
        KeldyshGF_2_components — Hartree-Fock contribution to the dual self-energy.
        """
        coeff = {"ch": 1, "sp": 3}
        Sigma_HF = KeldyshGF_2_components(self.t_mesh, target_shape=self.target_shape, is_reg=True)
        for channel in self.channels:
            Sigma_HF += (
                2 * coeff[channel]   # factor 2 from the spin sum over σ'
                * self.convolution2ndOrderHF(
                    Lambda[channel],
                    self.g,
                    self.w0[channel],  # use bare W̃⁰, not the dressed W̃
                    self.compute_f1_HF,
                    self.compute_f2_HF,
                    self.compute_sigma_HF,
                )
            )
        return Sigma_HF

    # ------------------------------------------------------------------
    # Dyson equation solvers
    # ------------------------------------------------------------------

    def computeFermionPropagator(self):
        """
        Update the dual GF G̃ by solving the fermionic Dyson equation (Eq. 26):

            (1 - G̃⁰ * Σ̃) * G̃ = G̃⁰

        where G̃⁰ is the bare dual GF (fixed throughout the iteration).
        Must be called after computeFermionSelfEnergy().
        """
        print("Updating fermionic propagator...")
        self.g = self.solveVie2(-self.g0 @ self.sigma, self.g0)
        herm_viol(self.g, "g")

    def computeBosonPropagators(self):
        """
        Update the dual bosonic propagator W̃ by solving the bosonic Dyson
        equation for each channel (Eqs. 28-29):

            W̃^δ remains unchanged  (Eq. 28)
            (1 - F̃(1,1̄)) * W̃^reg(1̄,2) = W̃^reg_0(1,2) + F̃(1,2) W̃^δ(2)

        where F̃ = W̃⁰ * Π̃.  The singular part W̃^δ = W̃⁰^δ is fixed.
        Must be called after computeBosonSelfEnergy().
        """
        print("Updating bosonic propagators...")
        for channel in self.channels:
            self.w[channel] = self.solveVie2(
                -self.w0[channel] @ self.pi[channel],
                self.w0[channel]
            )

            '''
            result_reg = self.w0[channel].reg
            result_reg += solve_vie2((-self.w0[channel] @ self.pi[channel]).reg,\
                                                                     (self.w0[channel] @ (-self.w0[channel] @ self.pi[channel]) @ self.w0[channel]).reg )
            self.w[channel] = KeldyshGF_2_components(
                self.t_mesh,
                self.target_shape,
                self.w0[channel].is_reg,
                sing=self.w0[channel].sing,
                reg=result_reg,
                )
            '''
            herm_viol(self.w[channel], f"w[{channel}]")

    # ------------------------------------------------------------------
    # Self-consistency iteration
    # ------------------------------------------------------------------

    def iterate(self, Lambda, prev_sigma: KeldyshGF_2_components,
                prev_g: KeldyshGF_2_components, prev_w, hartree_on=True):
        """
        Perform one full D-TRILEX iteration and return convergence errors.

        The iteration order follows the self-consistency loop described in the
        "D-TRILEX self-consistency loop" section of the notes:
            1. Compute Σ̃  from the current G̃ and W̃  (Eqs. 21/24)
            2. Update G̃   via Dyson equation           (Eq. 26)
            3. Compute Π̃  from the updated G̃           (Eq. 22/25)
            4. Update W̃   via Dyson equation           (Eqs. 28-29)

        Convergence is measured as the maximum absolute change in the regular
        part of each propagator compared to the previous iteration.

        Parameters
        ----------
        Lambda     : dict of KeldyshGF         — three-point vertices (fixed).
        prev_sigma : KeldyshGF_2_components    — Σ̃ from the previous iteration.
        prev_g     : KeldyshGF_2_components    — G̃ from the previous iteration.
        prev_w     : dict of KeldyshGF_2_components — W̃ from the previous iteration.
        hartree_on : bool — include the HF tadpole diagram.

        Returns
        -------
        (err_sigma, err_g, err_w_ch, err_w_sp) : floats
            Maximum absolute difference for Σ̃, G̃, W̃_ch, W̃_sp.
            Returns inf for the first iteration (no previous values to compare).
        """
        self.computeFermionSelfEnergy(Lambda, hartree_on)
        self.computeFermionPropagator()
        self.computeBosonSelfEnergy(Lambda)
        self.computeBosonPropagators()

        # Compute convergence errors (inf on the first call when prev_* are None).
        err_sigma = (
            float("inf")
            if prev_sigma is None
            else self.max_abs_keldyshgf(self.sigma.reg - prev_sigma.reg)
        )
        err_g = (
            float("inf")
            if prev_g is None
            else self.max_abs_keldyshgf(self.g.reg - prev_g.reg)
        )
        err_w_ch = (
            float("inf")
            if prev_w["ch"] is None
            else self.max_abs_keldyshgf(self.w["ch"].reg - prev_w["ch"].reg)
        )
        err_w_sp = (
            float("inf")
            if prev_w["sp"] is None
            else self.max_abs_keldyshgf(self.w["sp"].reg - prev_w["sp"].reg)
        )

        return err_sigma, err_g, err_w_ch, err_w_sp

    def max_abs_keldyshgf(self, G):
        """
        Return the maximum absolute value over all Keldysh components of G.

        Used as a simple sup-norm convergence metric: the iteration is
        considered converged when this value drops below the tolerance for
        all propagators.

        Parameters
        ----------
        G : KeldyshGF  — a two-point Keldysh Green's function.

        Returns
        -------
        float — max |G[branch1, branch2].data|.
        """
        if G is None:
            return 0.0
        mx = 0.0
        for a0, a1 in product(Branch, repeat=2):
            try:
                comp = G[a0, a1].data
                if isinstance(comp, (list, tuple)):
                    comp = np.array(comp)
                if comp.size:
                    mx = max(mx, float(np.max(np.abs(comp))))
            except Exception:
                pass  # skip Keldysh components that are not stored
        return mx


class DualQuantities2(DualQuantities):
    """
    D-TRILEX dual quantities for the Mickey-Mouse² model (Section 2).

    The Mickey-Mouse² system has two sites, so all propagators carry a 2×2
    site structure.  By the symmetry of the model, this matrix always has the
    block form  [[G_∥, G_⊥], [G_⊥, G_∥]]  (Eq. 52), so only the parallel
    (on-site, G_∥) and perpendicular (inter-site, G_⊥) components are stored
    as KeldyshGF_2x2_components objects.

    The key algorithmic trick is to diagonalise the 2×2 structure by going to
    the ± basis:
        G_+ = G_∥ + G_⊥,    G_- = G_∥ - G_⊥

    In this basis all Dyson equations decouple into two independent scalar
    equations (Eqs. 66-67, 74-75, 80-81), which are each solved by the parent
    class VIE2 solver.  The ∥/⊥ components are reconstructed afterwards.
    """

    def __init__(self, time_mesh, dual_params, delta_tilde, g_imp, pi_imp, Lambda):
        """
        Same parameters as DualQuantities.__init__, but delta_tilde and g_imp
        must be KeldyshGF_2x2_components objects.
        """
        self.t_mesh = time_mesh
        self.tt_mesh = MeshProduct(time_mesh, time_mesh)

        self.target_shape = (1, 1)
        
        self.channels = ["ch", "sp"]
        self.V = {"ch": dual_params["V"], "sp": 0.}  # inter-site interactions (same for both channels)
        self.V_sing = super().setW_bare(self.V)
        super().__init__(time_mesh, dual_params, delta_tilde, g_imp, pi_imp, Lambda)

    # ------------------------------------------------------------------
    # VIE2 in the diagonalised ± basis
    # ------------------------------------------------------------------

    def solveVie2(self, A: KeldyshGF_2x2_components, B: KeldyshGF_2x2_components):
        """
        Solve (1 - A) * X = B for a 2×2 propagator by diagonalising into ± sectors.

        The 2×2 Dyson equations (Eqs. 66-67, 74-75) decouple when written in
        terms of X_± = X_∥ ± X_⊥:
            (1 - A_±) * X_± = B_±,    A_± = A_∥ ± A_⊥

        Each ± equation is a scalar equation solved by the parent solveVie2.
        The ∥/⊥ components are recovered as:
            X_∥ = (X_+ + X_-) / 2,    X_⊥ = (X_+ - X_-) / 2   (Eqs. 68, 76)
        """
        X_plus  = super().solveVie2(A.par + A.perp, B.par + B.perp)
        X_minus = super().solveVie2(A.par - A.perp, B.par - B.perp)
        return KeldyshGF_2x2_components(
            par  = 0.5 * (X_plus + X_minus),
            perp = 0.5 * (X_plus - X_minus),
        )

    # ------------------------------------------------------------------
    # Convolutions in the diagonalised basis
    # ------------------------------------------------------------------

    def convolution2ndOrder(self, Lambda, g, w, compute_f1, compute_f2, compute):
        """
        Second-order diagram in the diagonalised basis for 2×2 propagators.

        For a bilinear diagram B(g, w) the diagonalisation identity gives:
            B(g_∥, w_∥) + B(g_⊥, w_⊥) = [B(g_+, w_+) + B(g_-, w_-)] / 2  → par
            B(g_∥, w_⊥) + B(g_⊥, w_∥) = [B(g_+, w_+) - B(g_-, w_-)] / 2  → perp

        (see the polarisation expressions Eqs. 59-60 and self-energy Eqs. 69-70).
        Falls back to the scalar parent method when g is not a 2×2 object.
        """
        if isinstance(g, KeldyshGF_2x2_components):
            res_plus  = super().convolution2ndOrder(
                Lambda, g.par + g.perp, w.par + w.perp, compute_f1, compute_f2, compute)
            res_minus = super().convolution2ndOrder(
                Lambda, g.par - g.perp, w.par - w.perp, compute_f1, compute_f2, compute)
            return KeldyshGF_2x2_components(
                par  = 0.5 * (res_plus + res_minus),
                perp = 0.5 * (res_plus - res_minus),
            )
        return super().convolution2ndOrder(Lambda, g, w, compute_f1, compute_f2, compute)

    def convolution2ndOrderHF(self, Lambda, g, w, compute_f1, compute_f2, compute):
        """
        Hartree-Fock diagram in the diagonalised basis for 2×2 propagators.

        Same diagonalisation as convolution2ndOrder; see Eqs. 69-70 (tadpole lines).
        """
        if isinstance(g, KeldyshGF_2x2_components):
            res_plus  = super().convolution2ndOrderHF(
                Lambda, g.par + g.perp, w.par + w.perp, compute_f1, compute_f2, compute)
            res_minus = super().convolution2ndOrderHF(
                Lambda, g.par - g.perp, w.par - w.perp, compute_f1, compute_f2, compute)
            return KeldyshGF_2x2_components(
                par  = 0.5 * (res_plus + res_minus),
                perp = 0.5 * (res_plus - res_minus),
            )
        return super().convolution2ndOrderHF(Lambda, g, w, compute_f1, compute_f2, compute)

    # ------------------------------------------------------------------
    # Overridden initialisers
    # ------------------------------------------------------------------

    def setGdual(self, g_imp):
        """
        Bare dual GF for the 2×2 model.  Solves Eq. (53):
            (1 - t_⊥ g t_⊥ g) * G̃_⊥ = t_⊥,    G̃_∥ = t_⊥ g * G̃_⊥
        encoded here as (1 - Δ̃ * g_imp) * G̃ = Δ̃ in 2×2 matrix form.
        """
        return self.solveVie2(-self.delta_tilde @ g_imp, self.delta_tilde)

    '''   
    def setW0(self, pi_imp):
        """
        Bare dual bosonic propagator W̃⁰ for the 2×2 model.

        The reference system is a single Mickey-Mouse unit, so its interaction
        is purely local (site-diagonal).  The inter-site (perp) component of
        W̃⁰ is therefore zero, and W̃⁰ is wrapped as a 2×2 object with
        par = scalar W̃⁰  and  perp = 0  (Eq. 58).
        """
        w0_base = super().setW0(pi_imp)
        result = {}
        for ch in self.channels:
            zero = KeldyshGF_2_components(
                self.t_mesh, self.target_shape, is_reg=w0_base[ch].is_reg
            )
            result[ch] = KeldyshGF_2x2_components(par=w0_base[ch], perp=zero)
        return result
    '''
    def setW0(self, pi_imp):
        """
        Compute the bare dual bosonic propagator W̃⁰ for each channel.

        From Eq. (17):
            W̃^δ_ς  = U^ς_sys - U^ς/2        (singular / contact part)
            W̃^reg_ς = W^reg_ς                 (regular part)

        where W^reg is the regular part of the reference bosonic propagator W_ς,
        obtained by solving (1 - U_sys * Π) * W^reg = U_sys * Π * U_sys  (Eq. 16).

        Parameters
        ----------
        U     : dict  — reference interactions per channel.
        U_sys : dict  — system interactions per channel.
        pi_imp: dict  — local polarisation Π_ς of the reference (Eq. 13).

        Returns
        -------
        dict of KeldyshGF_2_components, one per channel.
        """
        U_sys_keldysh = {}
        V_keldysh = {}
        U_keldysh = {}
        UV_keldysh = {}
        pi_keldysh = {}
        for channel in self.channels:
            U_keldysh[channel] = KeldyshGF_2x2_components(
                par =  KeldyshGF_2_components(
                    self.t_mesh, self.target_shape, is_reg=False, sing = self.w_sing[channel]),  # on-site interaction is the same as the scalar case
                perp = KeldyshGF_2_components(  # inter-site interaction is zero
                    self.t_mesh, self.target_shape, is_reg=True)
            )

            U_sys_keldysh[channel] = KeldyshGF_2x2_components(
                par =  KeldyshGF_2_components(
                    self.t_mesh, self.target_shape, is_reg=False, sing = self.w_sys_sing[channel]),  # on-site interaction is the same as the scalar case
                perp = KeldyshGF_2_components(  # inter-site interaction is zero
                    self.t_mesh, self.target_shape, is_reg=True)
            )

            V_keldysh[channel] = KeldyshGF_2x2_components(
                par =   KeldyshGF_2_components(  # inter-site interaction is zero
                    self.t_mesh, self.target_shape, is_reg=True), 
                perp = KeldyshGF_2_components(
                    self.t_mesh, self.target_shape, is_reg=False, sing = self.V_sing[channel]), 
            )

            UV_keldysh[channel] = U_sys_keldysh[channel] + V_keldysh[channel]

            pi_keldysh[channel] = KeldyshGF_2x2_components(
                par =  KeldyshGF_2_components(
                    self.t_mesh, self.target_shape, is_reg=True, reg = pi_imp[channel]),  # on-site interaction is the same as the scalar case
                perp = KeldyshGF_2_components(  # inter-site interaction is zero
                    self.t_mesh, self.target_shape, is_reg=True)
            )

        w0 = {}
        for channel in self.channels:
            w0[channel] = self.solveVie2(-UV_keldysh[channel] @ pi_keldysh[channel], UV_keldysh[channel])

            w0[channel] = w0[channel] - 0.5 * U_keldysh[channel]
        return w0

    # ------------------------------------------------------------------
    # Self-energy diagrams returning KeldyshGF_2x2_components
    # ------------------------------------------------------------------
    def computeGWDiagram(self, Lambda):
        """
        GW-like dual self-energy for the 2×2 model (Eqs. 69-70, GW-like lines).

        The parallel and perpendicular components of Σ̃ are obtained together
        via the diagonalised convolution2ndOrder.
        """
        coeff = {"ch": 1, "sp": 3}
        Sigma_GW = None
        for channel in self.channels:
            contrib = coeff[channel] * self.convolution2ndOrder(
                Lambda[channel], self.g, self.w[channel],
                self.compute_f1, self.compute_f2, self.compute_sigma,
            )
            Sigma_GW = contrib if Sigma_GW is None else Sigma_GW + contrib
        return Sigma_GW

    def computeHFDiagram(self, Lambda):
        """
        Hartree-Fock dual self-energy for the 2×2 model (Eqs. 69-70, tadpole lines).
        """
        coeff = {"ch": 1, "sp": 3}
        Sigma_HF = None
        for channel in self.channels:
            contrib = 2 * coeff[channel] * self.convolution2ndOrderHF(
                Lambda[channel], self.g, self.w0[channel],
                self.compute_f1_HF, self.compute_f2_HF, self.compute_sigma_HF,
            )
            Sigma_HF = contrib if Sigma_HF is None else Sigma_HF + contrib
        return Sigma_HF

    # ------------------------------------------------------------------
    # Self-consistency iteration with 2×2 convergence tracking
    # ------------------------------------------------------------------

    def iterate(self, Lambda, prev_sigma, prev_g, prev_w, hartree_on=True):
        """
        One D-TRILEX iteration for the 2×2 model.

        Identical in structure to DualQuantities.iterate (see Eqs. 69-70,
        59-60, 66-68, 74-76), but the convergence error is measured
        separately for the parallel and perpendicular components; the
        maximum of the two is returned.

        Returns
        -------
        (err_sigma, err_g, err_w_ch, err_w_sp) : floats — convergence errors.
        """
        self.computeFermionSelfEnergy(Lambda, hartree_on)
        self.computeFermionPropagator()
        self.computeBosonSelfEnergy(Lambda)
        self.computeBosonPropagators()

        # Use the parent's scalar max-norm on each ∥/⊥ component separately.
        _err = DualQuantities.max_abs_keldyshgf

        def reg_err(curr: KeldyshGF_2x2_components, prev):
            """Max error over both the parallel and perpendicular components."""
            if prev is None:
                return float("inf")
            return max(
                _err(self, curr.par.reg  - prev.par.reg),
                _err(self, curr.perp.reg - prev.perp.reg),
            )

        err_sigma = reg_err(self.sigma,     prev_sigma)
        err_g     = reg_err(self.g,         prev_g)
        err_w_ch  = reg_err(self.w["ch"],   prev_w.get("ch")  if prev_w else None)
        err_w_sp  = reg_err(self.w["sp"],   prev_w.get("sp")  if prev_w else None)

        return err_sigma, err_g, err_w_ch, err_w_sp


# ---------------------------------------------------------------------------
# Slicing helpers for time-stepping
# ---------------------------------------------------------------------------

def _slice_keldysh_gf(gf: KeldyshGF, n: int, t_mesh_n: MeshReTime) -> KeldyshGF:
    """
    Return a copy of `gf` restricted to the sub-mesh [t_0 .. t_n].

    All MeshReTime components of the original mesh are replaced by t_mesh_n.
    Non-time (e.g. BZ) mesh components are kept unchanged.
    """
    new_mesh_comps = [
        t_mesh_n if isinstance(m, MeshReTime) else m
        for m in gf.mesh.components
    ]
    new_mesh = MeshProduct(*new_mesh_comps)
    result = KeldyshGF(mesh=new_mesh, arg_index_shapes=gf.arg_index_shapes)

    # Slice the first n_args axes (time axes); keep everything else.
    slc = (slice(None, n + 1),) * gf.n_args + (Ellipsis,)
    for idx in np.ndindex(*((2,) * gf.n_args)):
        result.components[idx].data[:] = gf.components[idx].data[slc]
    return result


def _slice_singular_gf(
    sing: Singular2PKeldyshGF, n: int, t_mesh_n: MeshReTime
) -> Singular2PKeldyshGF:
    """
    Return a copy of `sing` restricted to the sub-mesh [t_0 .. t_n].
    """
    new_mesh = MeshProduct(t_mesh_n, *sing.non_time_mesh.components)
    result = Singular2PKeldyshGF(mesh=new_mesh, arg_index_shapes=sing.arg_index_shapes)
    slc = (slice(None, n + 1), Ellipsis)
    for b in range(2):
        result.components[b].data[:] = sing.components[b].data[slc]
    return result


def _slice_2_components(
    comp: KeldyshGF_2_components, n: int, t_mesh_n: MeshReTime
) -> KeldyshGF_2_components:
    """
    Return a copy of `comp` restricted to the sub-mesh [t_0 .. t_n].
    """
    return KeldyshGF_2_components(
        t_mesh_n,
        comp.target_shape,
        comp.is_reg,
        sing=_slice_singular_gf(comp.sing, n, t_mesh_n),
        reg=_slice_keldysh_gf(comp.reg, n, t_mesh_n),
    )



def _fill_old_times_2_components(
    dst: KeldyshGF_2_components, src: KeldyshGF_2_components
) -> None:
    """
    Overwrite the old time slices of `dst` in-place with values from `src`.

    `dst` lives on a mesh of size n+1 and `src` on size n (the previous step).
    All entries at time indices [0..n-1] are replaced by the converged values
    from `src`; the new time index n in `dst` is left untouched (keeps g0/w0).
    """
    n = src.reg.components.flat[0].data.shape[0]   # = old mesh size
    for idx in np.ndindex(*((2,) * dst.reg.n_args)):
        dst.reg.components[idx].data[:n, :n] = src.reg.components[idx].data[:, :]
    for b in range(2):
        dst.sing.components[b].data[:n] = src.sing.components[b].data[:]


# ---------------------------------------------------------------------------
# Time-stepping dual quantities
# ---------------------------------------------------------------------------

class DualQuantitiesTimestep:
    """
    D-TRILEX dual quantities solved step by step in real time.

    All reference quantities (Lambda, g_imp, pi_imp, delta_hyb) are assumed
    to be precomputed on the full time mesh.  At each timestep n the method
    `step(n)` slices them to the sub-mesh [t_0 .. t_n], builds a fresh
    DualQuantities object on that sub-mesh, and runs the full D-TRILEX
    self-consistency loop to convergence before returning.

    Because the Gregory quadrature requires at least `order + 1` time points,
    the minimum valid n is `KeldyshGF.integrator.order` (= 5 by default).
    """

    GREGORY_ORDER = KeldyshGF.integrator.order  # 5

    def __init__(
        self,
        t_mesh: MeshReTime,
        dual_params: dict,
        delta_hyb_full: KeldyshGF_2_components,
        g_imp_full: KeldyshGF_2_components,
        pi_imp_full: dict,
        Lambda_full: dict,
    ):
        """
        Parameters
        ----------
        t_mesh        : full real-time mesh shared by all propagators.
        dual_params   : interaction parameters (same as DualQuantities).
        delta_hyb_full: hybridisation difference Δ̃ on the full mesh.
        g_imp_full    : reference impurity GF on the full mesh.
        pi_imp_full   : dict {'ch', 'sp'} of KeldyshGF — local polarisation Π.
        Lambda_full   : dict {'ch', 'sp'} of KeldyshGF — three-point vertex Λ.
        """
        self.t_mesh = t_mesh
        self.dual_params = dual_params
        self.delta_hyb_full = delta_hyb_full
        self.g_imp_full = g_imp_full
        self.pi_imp_full = pi_imp_full
        self.Lambda_full = Lambda_full

        self.n_t = len(t_mesh)
        self._t_values = np.array([pt.value for pt in t_mesh])

    def _sub_mesh(self, n: int) -> MeshReTime:
        return MeshReTime(self._t_values[0], self._t_values[n], n + 1)

    def step(
        self,
        n: int,
        dual_g: KeldyshGF_2_components = None,
        dual_w: dict = None,
        max_iter: int = 20,
        tol: float = 1e-8,
        hartree_on: bool = False,
    ) -> DualQuantities:
        """
        Solve dual quantities on the sub-mesh [t_0 .. t_n] to convergence.

        Parameters
        ----------
        n          : timestep index; must be >= GREGORY_ORDER (= 5).
        dual_g     : converged G̃ from the previous step (warm start), or None.
        dual_w     : dict {'ch','sp'} of converged W̃ from the previous step, or None.
        max_iter   : maximum number of D-TRILEX self-consistency iterations.
        tol        : convergence threshold on max|ΔP| for all propagators.
        hartree_on : include the Hartree-Fock tadpole diagram.

        Returns
        -------
        DualQuantities converged on the sub-mesh of size n+1.
        """
        if n < self.GREGORY_ORDER:
            raise ValueError(
                f"n={n} < GREGORY_ORDER={self.GREGORY_ORDER}: "
                "sub-mesh too small for Gregory quadrature"
            )

        t_mesh_n = self._sub_mesh(n)

        delta_hyb_n = _slice_2_components(self.delta_hyb_full, n, t_mesh_n)
        g_imp_n     = _slice_2_components(self.g_imp_full,     n, t_mesh_n)
        pi_imp_n    = {
            ch: _slice_keldysh_gf(self.pi_imp_full[ch], n, t_mesh_n)
            for ch in ("ch", "sp")
        }
        Lambda_n    = {
            ch: _slice_keldysh_gf(self.Lambda_full[ch], n, t_mesh_n)
            for ch in ("ch", "sp")
        }

        dual_n = DualQuantities(
            t_mesh_n, self.dual_params, delta_hyb_n, g_imp_n, pi_imp_n, Lambda_n,
            init_g=dual_g, init_w=dual_w,
        )

        prev_sigma = None
        prev_g     = None
        prev_w     = {"ch": None, "sp": None}

        for it in range(max_iter):
            err_sigma, err_g, err_w_ch, err_w_sp = dual_n.iterate(
                Lambda_n, prev_sigma, prev_g, prev_w, hartree_on
            )
            print(
                f"  [n={n:3d}] iter {it + 1:2d}:"
                f" err_sigma={err_sigma:.3e}"
                f" err_g={err_g:.3e}"
                f" err_w_ch={err_w_ch:.3e}"
                f" err_w_sp={err_w_sp:.3e}"
            )
            prev_sigma   = dual_n.sigma
            prev_g       = dual_n.g
            prev_w["ch"] = dual_n.w["ch"]
            prev_w["sp"] = dual_n.w["sp"]

            if (
                err_sigma != float("inf")
                and err_sigma < tol
                and err_g     < tol
                and err_w_ch  < tol
                and err_w_sp  < tol
            ):
                print(f"  [n={n:3d}] Converged after {it + 1} iterations.")
                break

        return dual_n
