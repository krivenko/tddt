"""
Mickey-Mouse model classes for D-TRILEX on the Keldysh contour.

A Mickey-Mouse model is a minimal impurity system consisting of one
correlated site (on-site interaction U) coupled to one non-interacting
bath site.  Its non-interacting Hamiltonian is given by Eq. (1):

    H = -μ (n_↑ + n_↓) + U n_↑ n_↓  +  ε (n_↑,bath + n_↓,bath)
        + t (c†_↑ c_↑,bath + h.c. + c†_↓ c_↓,bath + h.c.)

This module provides two classes:
  - MickeyMouseModel  : single Mickey-Mouse unit used as the reference system.
  - MickeyMouseModel2 : two coupled Mickey-Mouse units (Mickey-Mouse²), the
                        full system described in Section 2 of the notes.
"""

from itertools import product
import numpy as np

from triqs.gf import Gf
from triqs.gf import MeshProduct
from realevol.tinterp import TInterp
from realevol.operators_tinterp import c, c_dag, n, a, a_dag
from programs.peierls import phase_tperp as _peierls_phase_tperp
from realevol.init_state import make_equilibrium_init_state

from programs.new_keldysh_classes import KeldyshGF_2_components
from tddt.keldysh import Branch, KeldyshGF, conv
from tddt.realevol import (
    compute_keldysh_gf,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3,
)

from programs.dual_quantities import KeldyshGF_2x2_components, DualQuantities2
from tddt.keldysh import herm_conj
from tddt.vie2 import solve_vie2


class MickeyMouseModel:
    """
    Single Mickey-Mouse impurity model (reference or system).

    This class encapsulates the full solution of one Mickey-Mouse unit:
    the real-time Green's function, charge/spin susceptibilities, the
    polarisation operator, the three-point vertex Λ, and the hybridisation
    function Δ = t · G_bath · t.

    After construction, call the compute* methods in order:
        1. computeG()          — full two-time GF on the Keldysh contour
        2. computeGimp()       — extract the impurity (site-0) block
        3. computeChiAndPi()   — susceptibility χ and polarisation Π
        4. computeLambda()     — three-point vertex Λ
    """

    def __init__(self, model_params, t_mesh):
        """
        Initialise the model and compute the bath hybridisation.

        Parameters
        ----------
        model_params : dict
            Physical parameters of the model:
              'U0'        — interaction before the quench (used for the initial state)
              'U'         — interaction after the quench (used for dynamics)
              'mu'        — chemical potential of the correlated site
              'eps_array' — 1-D array of bath site energies (length = number of bath sites)
              't_array'   — 1-D array of corr↔bath hoppings (same length as eps_array)
              'T'         — temperature for the initial equilibrium state
        t_mesh : MeshReTime
            Real-time mesh on which all propagators are evaluated.
        """
        self.t_mesh = t_mesh
        self.tt_mesh = MeshProduct(t_mesh, t_mesh)
        self.ttt_mesh = MeshProduct(t_mesh, t_mesh, t_mesh)

        self.U0 = model_params["U0"]
        self.U = model_params["U"]
        self.U_channel = {"ch": 0.5 * self.U, "sp": -0.5 * self.U}
        self.T = model_params["T"]
        self.mu = model_params["mu"]

        self.eps_array = np.asarray(model_params["eps_array"], dtype=float)
        self.t_array   = np.asarray(model_params["t_array"],   dtype=float)
        self.n_bath    = len(self.eps_array)

        self.flat_bath = model_params.get("flat_bath", False)
        self.D_flat    = model_params.get("D", None)
        if self.flat_bath:
            # ED uses only the correlated site; eps_array[0] / t_array[0]
            # are preserved for setBathFlat (they hold e0 and t_flat).
            self.n_bath = 0

        self.field_on = model_params.get("field_on", False)
        self.A0       = model_params.get("A0",    0.0)
        self.Omega    = model_params.get("Omega", 1.0)

        self.branches  = (Branch.FORWARD, Branch.BACKWARD)
        self.spin_names = ("up", "dn")

        # n_particles is the total Hilbert-space site count for this object.
        # MickeyMouseModel2 overrides _n_particles_total() to double it.
        self.n_particles = self._n_particles_total()
        self.fops = set(product(self.spin_names, range(self.n_particles)))

        self.hamiltonian = self.setHamiltonian(self.U0)
        self.init_state = self.setInitialState()
        self.hamiltonian = self.setHamiltonian(self.U, driven=True)

        # Hybridisation Δ = Σ_i t_i · G_i · t_i (summed over all bath sites).
        if self.flat_bath:
            self.hyb = self.setBathFlat()
        else:
            self.hyb = self.setBath()

        self.g_struct = [("up", self.n_particles), ("dn", self.n_particles)]

        self.g = None
        self.g_imp = None

        self.channels = ["ch", "sp"]
        self.chi_imp = None
        self.pi_imp = None
        self.Lambda = None

    def _n_particles_total(self):
        """Return the total number of sites in the Hilbert space for this model."""
        return 1 + self.n_bath

    def setHamiltonian(self, U, driven=False, corr_idx=0, bath_indices=None):
        """
        Build the second-quantised Hamiltonian for one Mickey-Mouse unit.

        Bath hoppings are always static — a Peierls phase on a single bath
        site can be gauged away and has no physical effect on the hybridisation
        or the correlated-site dynamics.  The `driven` flag is accepted for
        subclass compatibility (MickeyMouseModel2 uses it for t_⊥) but is
        ignored here.

        Parameters
        ----------
        U : float            — on-site Coulomb interaction.
        driven : bool        — ignored in base class; forwarded by MickeyMouseModel2.
        corr_idx : int       — site index of the correlated site. Default 0.
        bath_indices : list  — site indices of the bath sites. Default [1..n_bath].

        Returns
        -------
        operators_tinterp.Operator
        """
        if bath_indices is None:
            bath_indices = list(range(1, 1 + self.n_bath))
        i = corr_idx

        H = (
            -self.mu * (n("up", i) + n("dn", i))
            + U * n("up", i) * n("dn", i)
        )
        for j, eps_b, t_b in zip(bath_indices, self.eps_array, self.t_array):
            H += eps_b * (n("up", j) + n("dn", j))
            H += t_b * (
                c_dag("up", i) * c("up", j)
                + c_dag("up", j) * c("up", i)
                + c_dag("dn", i) * c("dn", j)
                + c_dag("dn", j) * c("dn", i)
            )
        return H

    def setInitialState(self):
        """
        Prepare the initial equilibrium density matrix at temperature T.

        The state is the grand-canonical equilibrium of the current
        self.hamiltonian (called with U0 during __init__ so the initial
        state corresponds to the pre-quench Hamiltonian).

        Returns
        -------
        InitState object used by the real-time evolution library.
        """
        init_state = make_equilibrium_init_state(
            self.hamiltonian,
            fermion_indices=self.fops,
            boson_indices=set(),
            temperature=self.T,
            params={},
        )
        return init_state

    def setBath(self):
        """
        Compute the total hybridisation Δ = Σ_i t_i · G_i · t_i over all bath sites.

        Occupation of each bath site is set by Fermi-Dirac at temperature T:
            occ_i = n_F(eps_i) = 1 / (exp(eps_i / T) + 1)

        For each level at energy ε_b with occupation occ_b:
            G^>(t1,t2) = -i (1 - occ_b) exp(-iε_b(t1-t2))
            G^<(t1,t2) = -i (-occ_b)    exp(-iε_b(t1-t2))

        Returns
        -------
        KeldyshGF  — total hybridisation Δ(t1,t2) on the two-time mesh.
        """
        hyb = None
        for eps_b, t_b in zip(self.eps_array, self.t_array):
            occ_b = 1.0 / (np.exp(eps_b / self.T) + 1.0)
            g_l = Gf(mesh=self.tt_mesh, target_shape=(1, 1))
            g_g = Gf(mesh=self.tt_mesh, target_shape=(1, 1))
            for time1, time2 in self.tt_mesh:
                phase = np.exp(-1j * eps_b * (float(time1) - float(time2)))
                g_g[time1, time2] = -1j * (1.0 - occ_b) * phase
                g_l[time1, time2] = -1j * (-occ_b) * phase
            g = KeldyshGF.from_lesser_greater(g_l, g_g)
            contrib = t_b * g * t_b
            hyb = contrib if hyb is None else hyb + contrib
        return hyb

    def setBathFlat(self):
        """Hybridisation from a flat-DOS bath: Δ = t² G_flat.

        FermionFlatBand.gf() returns a scalar KeldyshGF (target_shape=()).
        We wrap it in (1,1) shape so it is compatible with the discrete-bath
        hybridisation returned by setBath().
        """
        from tddt.models import FermionFlatBand
        from triqs.gf import Gf, MeshProduct
        e0     = float(self.eps_array[0])
        t_flat = float(self.t_array[0])
        ffb    = FermionFlatBand(self.D_flat, e0)
        g_scalar = ffb.gf(self.t_mesh, T=self.T)

        tt_mesh = MeshProduct(self.t_mesh, self.t_mesh)
        g_l = Gf(mesh=tt_mesh, target_shape=(1, 1))
        g_g = Gf(mesh=tt_mesh, target_shape=(1, 1))
        g_l.data[..., 0, 0] = g_scalar.lesser().data
        g_g.data[..., 0, 0] = g_scalar.greater().data
        g_bath = KeldyshGF.from_lesser_greater(g_l, g_g)
        return t_flat * g_bath * t_flat

    def computeGandGimp(self, params):
        """
        Compute the full two-time Green's function g_σ(t1, t2) on the Keldysh contour.

        This evaluates Eq. (2):
            g_σ(1,2) = -i <T c_σ(1) c†_σ(2)>

        for both spin components and all sites using exact real-time evolution.
        The result is stored in self.g as a dictionary keyed by spin ('up', 'dn'),
        each containing a KeldyshGF of shape (n_particles, n_particles).

        Parameters
        ----------
        params : dict
            Solver parameters passed to the real-time evolution library, e.g.
            verbosity, Hamiltonian interpolation scheme, Lanczos parameters.
        """
        self.g = compute_keldysh_gf(
            self.g_struct, self.init_state, self.hamiltonian, self.t_mesh, params
        )
        self.g_imp = self.computeGimp()

    def computeGimp(self, component=(0, 0)):
        """
        Extract the impurity (correlated site, site index 0) block from the full GF.

        The full GF self.g has shape (n_particles × n_particles) in site space.
        D-TRILEX only requires the (0,0) entry corresponding to the correlated site.
        The result is stored in self.g_imp as a KeldyshGF with target_shape=(1,1).

        Must be called after computeG().
        """
        g_imp = KeldyshGF(mesh=self.tt_mesh, target_shape=(1, 1))
        for br1 in self.branches:
            for br2 in self.branches:
                # Copy only the (0,0) site-space element into the 1×1 target slot.
                g_imp[br1, br2].data[..., 0, 0] = self.g["up"][br1, br2].data[
                    ..., component[0], component[1]
                ]
        return g_imp

    def computeChiAndPi(self, t_mesh, params):
        """
        Compute the local susceptibility χ_ς and polarisation Π_ς for each channel.

        The susceptibility is defined by Eq. (3):
            χ_ς(1,2) = -i <T b_ς(1) b_ς(2)>

        where b_ch = n_↑ + n_↓ (charge) and b_sp = n_↑ - n_↓ (spin).
        In practice χ is the connected correlator of the density operator.

        The polarisation Π_ς is obtained by solving the Dyson equation (Eq. 13):
            (1 + U^ς χ_ς) * Π_ς = χ_ς

        Results are stored in self.chi_imp and self.pi_imp as dicts keyed by
        channel ('ch' for charge, 'sp' for spin).

        Must be called after computeGimp().

        Parameters
        ----------
        t_mesh : MeshReTime  — time mesh for the correlator computation.
        params : dict        — solver parameters (same as in computeG).
        """
        # Density operators for charge and spin channels.
        ops = {"ch": n("up", 0) + n("dn", 0), "sp": n("up", 0) - n("dn", 0)}
        self.chi_imp = {}
        for channel in self.channels:
            # χ_ς = -i <T ρ_ς(t1) ρ_ς(t2)>_connected  (Eq. 3)
            self.chi_imp[channel] = -1j * compute_keldysh_conn_correlator_2t(
                ops[channel],
                ops[channel],
                self.init_state,
                self.hamiltonian,
                t_mesh,
                params,
            )

        self.pi_imp = {}
        for channel in self.channels:
            # Solve (1 + χ_ς U^ς) * Π_ς = χ_ς  →  Π_ς  (Eq. 13)
            susc_U = self.chi_imp[channel] * self.U_channel[channel]
            self.pi_imp[channel] = solve_vie2(susc_U, self.chi_imp[channel])

            # Reshape Π to carry explicit (1,1) target axes so it is compatible
            # with the Singular2PKeldyshGF objects used elsewhere.
            orig_pi = self.pi_imp[channel]
            desired_arg_shapes = ((1,), (1,))
            if getattr(orig_pi, "arg_index_shapes", None) != desired_arg_shapes:
                new_pi = KeldyshGF(
                    mesh=orig_pi.mesh, arg_index_shapes=desired_arg_shapes
                )
                for a0, a1 in product(Branch, repeat=2):
                    comp = orig_pi[a0, a1].data
                    new_pi[a0, a1].data[:] = comp.reshape(comp.shape + (1, 1))
                self.pi_imp[channel] = new_pi

    def computeLambda(self, t_mesh, params):
        """
        Compute the three-point vertex Λ^ς for each channel.

        Implements Eq. (10):
            Λ^ς(1,2,3) = -∫d3' α^{-1}_ς(3,3') <T c(1) c*(2) ρ_ς(3')>

        which in the D-TRILEX parametrisation (B = δ, α = δ + U^ς χ_ς, Eqs. 8-9)
        simplifies to:
            Λ^ς = -<T c c* ρ_ς> + <T c c* ρ_ς> * U^ς * Π_ς

        Only the up-up spin component is computed (the paramagnetic assumption
        Λ^{↑↑} = Λ^{↓↓} is used, as stated below Eq. 10).

        The result is stored in self.Lambda as a dict keyed by channel.

        Must be called after computeChiAndPi().

        Parameters
        ----------
        t_mesh : MeshReTime  — time mesh.
        params : dict        — solver parameters.
        """
        if self.Lambda is None:
            ops = {"ch": n("up", 0) + n("dn", 0), "sp": n("up", 0) - n("dn", 0)}
            self.Lambda = {}
            for channel in self.channels:
                # Compute the bare three-point correlator <T c_↑(1) c†_↑(2) ρ_ς(3)>.
                three_point_corr = compute_keldysh_vertex3(
                    ("up", 0),
                    ("up", 0),
                    ops[channel],
                    self.init_state,
                    self.hamiltonian,
                    t_mesh,
                    params,
                )

                # Reshape to explicit (1,1,1) target axes for convolution compatibility.
                desired_3pt_arg_shapes = ((1,), (1,), (1,))
                if (
                    getattr(three_point_corr, "arg_index_shapes", None)
                    != desired_3pt_arg_shapes
                ):
                    new_3pt = KeldyshGF(
                        mesh=three_point_corr.mesh,
                        arg_index_shapes=desired_3pt_arg_shapes,
                    )
                    for a0, a1, a2 in product(Branch, repeat=3):
                        comp = three_point_corr[a0, a1, a2].data
                        new_3pt[a0, a1, a2].data[:] = comp.reshape(
                            comp.shape + (1, 1, 1)
                        )
                    three_point_corr = new_3pt

                # Λ^ς = -<Tcc*ρ> + <Tcc*ρ> * (U^ς * Π_ς)   (Eq. 10)
                U_pi = self.U_channel[channel] * self.pi_imp[channel]
                three_point_corr_U_pi = conv(three_point_corr, U_pi, [(2, 1)])
                self.Lambda[channel] = -three_point_corr + three_point_corr_U_pi

    def apply_eta_damping(self, eta):
        """
        Multiply all impurity quantities by the adiabatic envelope e^{-η(t1+t2+...)}
        to suppress secular growth in the dual self-energy.

        The damping cuts the effective integration domain: an integrand that
        would otherwise contribute up to t_max decays to ~e^{-η*t_max} at the
        boundary. This is a controlled approximation equivalent to adding a
        small imaginary part to the Hamiltonian (finite lifetime broadening).

        Must be called after computeGandGimp(), computeChiAndPi(), and
        computeLambda(), and before building dual quantities.

        Parameters
        ----------
        eta : float
            Damping rate. eta=0 is a no-op. Typical values: 0.05–0.1.
        """
        if eta == 0.0:
            return

        t_values = np.array([t.value for t in self.t_mesh])
        damp1 = np.exp(-eta * t_values * t_values)

        def _apply(gf, n_time_axes):
            """Multiply each Keldysh component of gf by the n_time_axes-fold envelope."""
            # Build damping array with shape (n_t,)*n_time_axes.
            damp = damp1.copy()
            for _ in range(n_time_axes - 1):
                damp = np.multiply.outer(damp, damp1)
            for br in product(Branch, repeat=n_time_axes):
                d = gf[br].data
                # Broadcast over any trailing target-space axes.
                damp_bc = damp.reshape(damp.shape + (1,) * (d.ndim - n_time_axes))
                d *= damp_bc

        _apply(self.g_imp, 2)
        for ch in self.channels:
            _apply(self.chi_imp[ch], 2)
            _apply(self.pi_imp[ch], 2)
            _apply(self.Lambda[ch], 3)

    def computeFermionPropagatorFromDual(self, delta_hyb: KeldyshGF, sigma_dual=None):
        """
        Recover the physical impurity Green's function G from the dual self-energy.

        After the D-TRILEX self-consistency is converged, the corrected physical
        GF is obtained from the Dyson equation (see Physical Green's function section,
        equivalent to Eqs. 80-82 for the scalar case):

            (1 - K * Δ̃) * G = K,    K = g_imp + Σ̃

        where Σ̃ is the dual self-energy and Δ̃ = Δ_sys - Δ_ref is the hybridisation
        difference.  The right-hand side K is symmetrised to enforce hermiticity.

        Parameters
        ----------
        delta_hyb : KeldyshGF
            Regular part of the hybridisation difference Δ̃ = Δ_sys - Δ_ref (Eq. 11).
        sigma_dual : KeldyshGF
            Regular part of the dual fermionic self-energy Σ̃ (Eq. 23).

        Returns
        -------
        KeldyshGF  — the corrected physical impurity Green's function G.
        """
        print("Recovering physical GF from dual self-energy...")
        K = self.g_imp + sigma_dual if sigma_dual is not None else self.g_imp
        Q = 0.5 * (K + herm_conj(K))  # symmetrise to enforce G = G†
        return solve_vie2(-K @ delta_hyb, Q)

    def computeFermionPropagatorFromDual2(
        self,
        delta_hyb: KeldyshGF_2x2_components,
        dual_quantities: DualQuantities2,
        g_imp: KeldyshGF_2x2_components,
        is_cpt: bool = False,
    ):
        """
        Recover the physical GF for the Mickey-Mouse² model.

        The 2×2 analogue of computeFermionPropagatorFromDual, solving
        Eqs. (80-82) in the parallel/perpendicular basis.  The Dyson equation
        is solved via the diagonal-basis trick of DualQuantities2.solveVie2.

        Parameters
        ----------
        delta_hyb : KeldyshGF_2x2_components
            Hybridisation difference matrix Δ̃ with parallel and perpendicular
            components (Eq. 51, 53).
        dual_quantities : DualQuantities2
            The converged dual-quantities object holding the dual self-energy Σ̃.
        g_imp : KeldyshGF_2x2_components
            Reference impurity GF wrapped in 2×2 structure (G̃_par = g_imp, G̃_perp = 0).

        Returns
        -------
        KeldyshGF_2x2_components — physical GF with parallel and perpendicular components.
        """
        K = (
            g_imp + dual_quantities.sigma if not is_cpt else g_imp
        )  # CPT uses g_imp without Σ̃ correction
        return dual_quantities.solveVie2(-K @ delta_hyb, K)


class MickeyMouseModel2(MickeyMouseModel):
    """
    Mickey-Mouse² model: two coupled Mickey-Mouse units.

    The full system consists of two identical Mickey-Mouse models connected by
    an inter-site hopping t_⊥ and an inter-site Coulomb interaction V
    (see Figure 1 and Section 2 of the notes).  The Hamiltonian is:

        H = H_site1(U) + H_site2(U)
            + t_⊥ (c†_↑,0 c_↑,2 + h.c. + c†_↓,0 c_↓,2 + h.c.)
            + (V/2) (n_0 + n_1)(n_2 + n_3)          [n_i = n_↑,i + n_↓,i]

    The reference system for the dual transformation is a single Mickey-Mouse
    unit (MickeyMouseModel), and the system has sites 0,1 (first unit) and
    2,3 (second unit).
    """

    def __init__(self, model_params, t_mesh):
        """
        Parameters
        ----------
        model_params : dict
            Same keys as MickeyMouseModel, plus:
              't_perp' — inter-site hopping between the two correlated sites.
              'V'      — inter-site Coulomb interaction.
        t_mesh : MeshReTime
            Real-time mesh.
        """
        # V, t_perp, and n_bath must be set before super().__init__ because the
        # parent constructor calls setHamiltonian() and _n_particles_total(), both
        # of which need these attributes.
        self.V      = model_params["V"]
        self.t_perp = model_params["t_perp"]
        # n_bath is needed by _n_particles_total before super().__init__ sets it.
        self.n_bath = len(model_params["eps_array"])
        super().__init__(model_params, t_mesh)

    def _n_particles_total(self):
        return 2 * (1 + self.n_bath)

    def setHamiltonian(self, U, driven=False, **_):
        """
        Build the Hamiltonian for both Mickey-Mouse units plus inter-site coupling.

        Site layout (N = 1 + n_bath sites per unit):
          unit 1: correlated=0,  bath=[1 .. N-1]
          unit 2: correlated=N,  bath=[N+1 .. 2N-1]

        Parameters
        ----------
        U : float     — on-site interaction (same for both units).
        driven : bool — passed through to the base class and applied to t_⊥.

        Returns
        -------
        operators_tinterp.Operator  (static) or
        np.ndarray of operators_tinterp.Operator, shape (n_t,)  (driven).
        """
        N = 1 + self.n_bath
        i0, i1 = 0, N
        bath1 = list(range(1, N))
        bath2 = list(range(N + 1, 2 * N))
        h1 = super().setHamiltonian(U, driven=driven, corr_idx=i0, bath_indices=bath1)
        h2 = super().setHamiltonian(U, driven=driven, corr_idx=i1, bath_indices=bath2)

        if driven and self.field_on and self.A0 != 0.0:
            t_arr = np.array([float(t) for t in self.t_mesh])
            ph_arr = _peierls_phase_tperp(t_arr, self.A0, self.Omega)
            ph_ti      = TInterp(self.t_mesh, ph_arr)
            ph_conj_ti = TInterp(self.t_mesh, np.conj(ph_arr))
            int_hop = (self.t_perp * ph_ti) * (
                c_dag("up", i0) * c("up", i1) + c_dag("dn", i0) * c("dn", i1)
            ) + (self.t_perp * ph_conj_ti) * (
                c_dag("up", i1) * c("up", i0) + c_dag("dn", i1) * c("dn", i0)
            )
        else:
            int_hop = self.t_perp * (
                c_dag("up", i0) * c("up", i1)
                + c_dag("up", i1) * c("up", i0)
                + c_dag("dn", i0) * c("dn", i1)
                + c_dag("dn", i1) * c("dn", i0)
            )

        # V couples total density of unit 1 to total density of unit 2.
        # Density-density interaction carries no Peierls phase.
        n_unit1 = sum(n("up", s) + n("dn", s) for s in range(N))
        n_unit2 = sum(n("up", s) + n("dn", s) for s in range(N, 2 * N))
        int_coulomb = 0.5 * self.V * n_unit1 * n_unit2
        return h1 + h2 + int_hop + int_coulomb

    def computeGimp(self):
        N = 1 + self.n_bath
        g_imp_par = super().computeGimp((0, 0))
        g_imp_NN  = super().computeGimp((N, N))
        for br1 in self.branches:
            for br2 in self.branches:
                diff = np.max(np.abs(
                    g_imp_par[br1, br2].data[..., 0, 0]
                    - g_imp_NN[br1, br2].data[..., 0, 0]
                ))
                print(f"[Mickey2.computeGimp] G(0,0) vs G({N},{N})  branch ({br1},{br2})  max|diff| = {diff:.3e}")
        g_imp_perp = super().computeGimp((0, N))
        g_imp = KeldyshGF_2x2_components(
            par=KeldyshGF_2_components(
                self.t_mesh,
                target_shape=(1, 1),
                is_reg=True,
                reg=g_imp_par,
            ),
            perp=KeldyshGF_2_components(
                self.t_mesh,
                target_shape=(1, 1),
                is_reg=True,
                reg=g_imp_perp,
            ),
        )
        return g_imp
