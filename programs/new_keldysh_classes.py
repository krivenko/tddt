"""
Wrapper classes for Keldysh Green's functions used in D-TRILEX.

Both classes represent the decomposition of a propagator into a singular
(instantaneous / contact) part and a regular (smooth, two-time) part,
as described in Eq. (15) of the notes:

    W(1,2) = W^δ(1) δ(1,2) + W^reg(1,2)

This separation is necessary because the bare dual bosonic propagator W̃
contains a contact term proportional to the interaction U, which must be
treated analytically to avoid numerical artefacts.

KeldyshGF_2_components  — single-site propagator (scalar in site space).
KeldyshGF_2x2_components — two-site propagator with the block symmetry
    G = [[G_par, G_perp], [G_perp, G_par]]
that arises in the Mickey-Mouse² model (Eq. 52).
"""

import copy
import numpy as np

from triqs.gf import MeshProduct

from tddt.keldysh import KeldyshGF, Branch
from tddt.keldysh import Singular2PKeldyshGF


def _sing_times_reg(G_sing: Singular2PKeldyshGF, G_reg: KeldyshGF) -> KeldyshGF:
    """
    Compute (f·δ_C @ G_reg)(t,t') = f(t) · G_reg(t,t') exactly.

    Avoids the generic conv() path which applies Gregory quadrature to a
    δ-function integral and introduces O(dt^p) hermiticity violation.
    Works for time-dependent f(t).
    """
    result = copy.deepcopy(G_reg)
    for b1 in (Branch.FORWARD, Branch.BACKWARD):
        for b2 in (Branch.FORWARD, Branch.BACKWARD):
            # G_sing[b1].data  : (n_t, *contracted, *right)
            # G_reg[b1,b2].data: (n_t, n_t, *left, *contracted)
            # result           : (n_t, n_t, *left, *right)
            result[b1, b2].data[...] = np.einsum(
                'ikl,ijlm->ijkm',
                G_sing[b1].data,
                G_reg[b1, b2].data,
            )
    return result


def _reg_times_sing(G_reg: KeldyshGF, G_sing: Singular2PKeldyshGF) -> KeldyshGF:
    """
    Compute (G_reg @ f·δ_C)(t,t') = G_reg(t,t') · f(t') exactly.

    Avoids the generic conv() path which applies Gregory quadrature to a
    δ-function integral and introduces O(dt^p) hermiticity violation.
    Works for time-dependent f(t).
    """
    result = copy.deepcopy(G_reg)
    for b1 in (Branch.FORWARD, Branch.BACKWARD):
        for b2 in (Branch.FORWARD, Branch.BACKWARD):
            # G_reg[b1,b2].data: (n_t, n_t, *left, *contracted)
            # G_sing[b2].data  : (n_t, *contracted, *right)
            # result           : (n_t, n_t, *left, *right)
            result[b1, b2].data[...] = np.einsum(
                'ijkl,jlm->ijkm',
                G_reg[b1, b2].data,
                G_sing[b2].data,
            )
    return result


def _lambda_times_sing(Lambda: KeldyshGF, G_sing: Singular2PKeldyshGF) -> KeldyshGF:
    """
    Compute F2(t1,t2,t3) = Lambda(t1,t2,t3) · f(t3) exactly.

    This is the exact result of conv(Lambda, f·δ_C, [(2,1)], free_args=([0,1],[2])):
    contracting Lambda's 3rd time argument with f·δ_C's 1st argument gives
    Lambda(t1,t2,t3) · f(t3) — no quadrature needed.

    Avoids the generic conv() path which applies Gregory quadrature to what is
    analytically a δ-function integral, introducing O(dt^p) hermiticity violation
    in sigma via compute_f2(Lambda, w.sing).
    """
    result = copy.deepcopy(Lambda)
    for b1 in (Branch.FORWARD, Branch.BACKWARD):
        for b2 in (Branch.FORWARD, Branch.BACKWARD):
            for b3 in (Branch.FORWARD, Branch.BACKWARD):
                # Lambda[b1,b2,b3].data: (n_t, n_t, n_t, *target_abc)
                # G_sing[b3].data:       (n_t,           *target_cd)
                # result:                (n_t, n_t, n_t, *target_abd)
                result[b1, b2, b3].data[...] = np.einsum(
                    'ijkabc,kcd->ijkabd',
                    Lambda[b1, b2, b3].data,
                    G_sing[b3].data,
                )
    return result


def _sing_times_lambda(G_sing: Singular2PKeldyshGF, Lambda: KeldyshGF) -> KeldyshGF:
    """
    Compute F(t1,t2,t3) = f(t1) · Lambda(t1,t2,t3) exactly.

    Analogous to _lambda_times_sing but contracts Lambda's 1st time argument
    with f·δ_C's argument, yielding f(t1) · Lambda(t1,t2,t3) — no quadrature needed.

    Avoids the generic conv() path which applies Gregory quadrature to what is
    analytically a δ-function integral, introducing O(dt^p) hermiticity violation.
    """
    result = copy.deepcopy(Lambda)
    for b1 in (Branch.FORWARD, Branch.BACKWARD):
        for b2 in (Branch.FORWARD, Branch.BACKWARD):
            for b3 in (Branch.FORWARD, Branch.BACKWARD):
                # Lambda[b1,b2,b3].data: (n_t, n_t, n_t, *target_abc)
                # G_sing[b1].data:       (n_t,           *target_cd)
                # result:                (n_t, n_t, n_t, *target_abd)
                result[b1, b2, b3].data[...] = np.einsum(
                    'ijkabc,icd->ijkabd',
                    Lambda[b1, b2, b3].data,
                    G_sing[b1].data,
                )
    return result


class KeldyshGF_2_components:
    """
    A Keldysh Green's function stored as the sum of a singular and a regular part.

    Following Eq. (15), any propagator P can be written as:

        P(t1, t2) = P^sing(t1) * δ(t1 - t2)  +  P^reg(t1, t2)

    where:
      - sing (Singular2PKeldyshGF) carries the instantaneous (δ-function) piece,
        e.g. the bare interaction U or the bare dual W̃^δ (Eq. 17/58).
      - reg  (KeldyshGF)           carries the smooth two-time piece.

    The flag `is_reg = True` means sing is identically zero, i.e. the
    propagator has no contact term.  This is the case for the hybridisation
    Δ and for the impurity Green's function g_imp.

    All arithmetic operators (+, -, *, @) are overloaded so that objects of
    this class can be combined with standard Python syntax.  The @ operator
    implements the Keldysh convolution P @ Q, keeping track of all
    cross-terms between singular and regular parts.
    """

    def __init__(self, t_mesh, target_shape=None, is_reg: bool = False, sing=None, reg=None):
        """
        Parameters
        ----------
        t_mesh : MeshReTime
            The one-dimensional real-time mesh.
        target_shape : tuple, optional
            Shape of the orbital/site target space, e.g. (1, 1).  Defaults to (1, 1).
        is_reg : bool, optional
            Set True if the propagator has no singular (contact) part.
        sing : Singular2PKeldyshGF, optional
            The instantaneous part.  If None, initialised to zero.
        reg : KeldyshGF, optional
            The regular two-time part.  If None, initialised to zero.
        """
        self.target_shape = target_shape if target_shape is not None else (1, 1)

        self.t_mesh = t_mesh
        self.tt_mesh = MeshProduct(t_mesh, t_mesh)
        self.is_reg = is_reg

        # Initialise singular part to zero if not provided.
        self.sing = (
            sing
            if sing is not None
            else Singular2PKeldyshGF(mesh=self.t_mesh, target_shape=self.target_shape)
        )
        # Initialise regular part to zero if not provided.
        self.reg = (
            reg
            if reg is not None
            else KeldyshGF(mesh=self.tt_mesh, target_shape=self.target_shape)
        )

    def __add__(self, other):
        """Component-wise addition: (A + B).sing = A.sing + B.sing, same for reg.
        Result is_reg only if both operands have no singular part."""
        x = KeldyshGF_2_components(self.t_mesh, self.target_shape, self.is_reg)
        x.is_reg = self.is_reg and other.is_reg
        x.sing = self.sing + other.sing
        x.reg = self.reg + other.reg
        return x

    def __neg__(self):
        """Unary minus: negate both components."""
        x = KeldyshGF_2_components(self.t_mesh, self.target_shape, self.is_reg)
        x.sing = -self.sing
        x.reg = -self.reg
        return x

    def __sub__(self, other):
        """Component-wise subtraction."""
        x = KeldyshGF_2_components(self.t_mesh, self.target_shape, self.is_reg)
        x.is_reg = self.is_reg and other.is_reg
        x.sing = self.sing - other.sing
        x.reg = self.reg - other.reg
        return x

    def __mul__(self, scalar):
        """Multiply both components by a scalar: scalar * P."""
        x = KeldyshGF_2_components(self.t_mesh, self.target_shape, self.is_reg)
        x.sing = scalar * self.sing
        x.reg = scalar * self.reg
        return x

    def __rmul__(self, scalar):
        """Right-multiply by scalar (same as __mul__ for commuting scalars)."""
        x = KeldyshGF_2_components(self.t_mesh, self.target_shape, self.is_reg)
        x.sing = scalar * self.sing
        x.reg = scalar * self.reg
        return x

    def __matmul__(self, other):
        """
        Keldysh convolution A @ B, expanding all singular/regular cross-terms:

            (A @ B)^sing = A^sing @ B^sing
            (A @ B)^reg  = A^sing @ B^reg + A^reg @ B^sing + A^reg @ B^reg

        The result has a non-zero singular part only when both A and B do,
        so is_reg is True (result is purely regular) if either operand is.
        """
        x = KeldyshGF_2_components(self.t_mesh, self.target_shape, self.is_reg)
        # If either operand is purely regular, the singular cross-term vanishes.
        x.is_reg = self.is_reg or other.is_reg
        x.sing = self.sing @ other.sing
        x.reg = (
            _sing_times_reg(self.sing, other.reg)   # exact: no quadrature on δ
            + _reg_times_sing(self.reg, other.sing)  # exact: no quadrature on δ
            + self.reg @ other.reg
        )
        return x


class KeldyshGF_2x2_components:
    """
    A two-site Keldysh Green's function with the block symmetry of the
    Mickey-Mouse² model (Eq. 52 of the notes):

        G = [[ G_par,  G_perp ],
             [ G_perp, G_par  ]]

    Only the two independent components are stored:
      - par  (KeldyshGF_2_components): on-site (parallel) block G_∥
      - perp (KeldyshGF_2_components): inter-site (perpendicular) block G_⊥

    This symmetry is preserved by all arithmetic operations defined here.
    The @ operator implements the 2×2 matrix convolution, using the rule:

        (A @ B)_par  = A_par @ B_par + A_perp @ B_perp
        (A @ B)_perp = A_par @ B_perp + A_perp @ B_par
    """

    def __init__(self, par: KeldyshGF_2_components, perp: KeldyshGF_2_components):
        """
        Parameters
        ----------
        par  : KeldyshGF_2_components  — on-site (diagonal) component G_∥
        perp : KeldyshGF_2_components  — inter-site (off-diagonal) component G_⊥
        """
        self.par = par
        self.perp = perp

    @property
    def is_reg(self):
        """True if both par and perp have no singular part."""
        return self.par.is_reg and self.perp.is_reg

    def __add__(self, other):
        return KeldyshGF_2x2_components(self.par + other.par, self.perp + other.perp)

    def __neg__(self):
        return KeldyshGF_2x2_components(-self.par, -self.perp)

    def __sub__(self, other):
        return KeldyshGF_2x2_components(self.par - other.par, self.perp - other.perp)

    def __mul__(self, scalar):
        return KeldyshGF_2x2_components(scalar * self.par, scalar * self.perp)

    def __rmul__(self, scalar):
        return KeldyshGF_2x2_components(scalar * self.par, scalar * self.perp)

    def __matmul__(self, other):
        """
        2×2 block matrix convolution, exploiting the [[par, perp],[perp, par]] symmetry:

            (A @ B)_par  = A_par @ B_par  + A_perp @ B_perp
            (A @ B)_perp = A_par @ B_perp + A_perp @ B_par
        """
        return KeldyshGF_2x2_components(
            par=self.par @ other.par + self.perp @ other.perp,
            perp=self.par @ other.perp + self.perp @ other.par,
        )
