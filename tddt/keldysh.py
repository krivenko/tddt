#
# Keldysh Green's functions and vertices
#


from enum import Enum
from copy import deepcopy
from itertools import product, takewhile, islice, chain
from typing import Tuple, Dict, Union, Sequence, Optional, Callable
import numpy as np

from triqs.gf import Gf, MeshReTime, MeshPoint, MeshProduct

from .util import subscripts
from .integration import GregoryIntegrator
from .retime import conv_ret_lg, conv_lg_adv


class Branch(Enum):
    """Branch of the Keldysh contour"""
    FORWARD = 0
    BACKWARD = 1


class ContourPoint:
    """
    Point on the Keldysh contour, combination of a branch and a real time point
    """

    def __init__(self, branch, t):
        assert isinstance(t, MeshPoint)
        self.branch = branch
        self.t = t

    def __lt__(self, other):
        """
        This function defines the comparison rule used by `contour_ordering()`.
        """
        if self.branch == other.branch:
            if self.branch == Branch.FORWARD:
                return self.t.linear_index < other.t.linear_index
            else:
                return self.t.linear_index >= other.t.linear_index
        else:
            return self.branch.value < other.branch.value


def contour_ordering(*points):
    """
    Contour ordering of a list of points

    Takes a list of N contour points and returns a permutation of integers
    (0, 1, N-1) describing the order of the points on the contour. A pair of
    coinciding points on the forward branch comes in the original order in the
    output permutation, while for the backward branch the order is reversed.
    """
    return tuple(sorted(range(len(points)),
                        key=lambda n: points[n],
                        reverse=True))


class KeldyshGF:
    """Generic N-point Green's function defined on a 2-branch Keldysh contour"""

    """Integrator object for contour convolutions"""
    integrator = GregoryIntegrator(5)

    def __init__(
            self, *,
            mesh: Union[MeshReTime, MeshProduct],
            target_shape: Optional[Tuple[int, ...]] = None,
            arg_index_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None):

        #
        # Process the supplied mesh
        #

        if isinstance(mesh, MeshReTime):  # Single-argument contour function
            self.mesh = MeshProduct(mesh)
            self.time_mesh = self.mesh
            self.non_time_mesh = MeshProduct()
            self.n_args = 1

        elif isinstance(mesh, MeshProduct):  # N-point Green's function
            self.mesh = mesh
            self.time_mesh = MeshProduct(
                *takewhile(lambda m: isinstance(m, MeshReTime), mesh.components)
            )
            self.n_args = len(self.time_mesh.components)
            if len(mesh.components[self.n_args:]) != 0:
                self.non_time_mesh = MeshProduct(*mesh.components[self.n_args:])
            else:
                self.non_time_mesh = MeshProduct()
        else:
            raise TypeError(f"Unsupported mesh type {type(mesh)}")

        #
        # Process the target subshapes
        #

        # All subshapes are 0-dimensional by default
        if (target_shape is None) and (arg_index_shapes is None):
            self.arg_index_shapes = ((),) * self.n_args
            self.target_shape = ()

        elif (target_shape is None) and (arg_index_shapes is not None):
            assert len(arg_index_shapes) == self.n_args, \
                f"arg_index_shapes must contain {self.n_args} elements for " \
                f"a {self.n_args}-point function"
            self.arg_index_shapes = tuple(arg_index_shapes)
            self.target_shape = sum(arg_index_shapes, ())

        elif (target_shape is not None) and (arg_index_shapes is None):
            assert len(target_shape) % self.n_args == 0, \
                f"Target shape must have a multiple of {self.n_args} elements"
            self.target_shape = tuple(target_shape)
            tss_len = len(self.target_shape) // self.n_args
            self.arg_index_shapes = tuple(
                self.target_shape[i:i + tss_len]
                for i in range(0, len(self.target_shape), tss_len)
            )

        else:
            raise RuntimeError(
                "target_shape and arg_index_shapes are mutually exclusive"
            )

        #
        # Allocate data storage
        #

        self.components = np.array(
            [Gf(mesh=self.mesh, target_shape=self.target_shape)
             for _ in range(2 ** self.n_args)]
        ).reshape((2,) * self.n_args)

    @classmethod
    def from_arg_index_gen(cls,
                           generator: Callable, *,
                           mesh: Union[MeshReTime, MeshProduct],
                           arg_index_shapes: Tuple[Tuple[int, ...], ...]
                           ):
        """
        Construct a matrix/tensor-valued KeldyshGF object out of scalar-valued
        components returned by a given generator function.

        The generator function used to construct an N-point GF must accept N
        arguments, each of which is a tuple of integer indices from the ranges
        determined by the respective parts of `arg_index_shapes`. The generator
        is expected to return a scalar-valued KeldyshGF object.
        """
        g = KeldyshGF(mesh=mesh, arg_index_shapes=arg_index_shapes)

        for indices in product(*map(np.ndindex, arg_index_shapes)):
            g_el = generator(*indices)
            assert isinstance(g_el, KeldyshGF), \
                "Generator must return an instance of KeldyshGF"
            assert g.n_args == g_el.n_args, \
                f"Expected a {g.n_args}-point GF from the generator " \
                f"(got a {g_el.n_args}-point GF)"
            assert g.mesh == g_el.mesh, \
                "GF returned by the generator has a wrong mesh"

            for g_comp, g_el_comp in zip(g.components.flat,
                                         g_el.components.flat):
                sl = (slice(None),) * len(mesh.components) + \
                    tuple(chain.from_iterable(indices))
                g_comp.data[sl] = g_el_comp.data

        return g

    def __getitem__(self, args):
        args_t = args if isinstance(args, tuple) else (args,)

        assert len(args_t) >= self.n_args, \
            f"At least {self.n_args} arguments are required"

        if all(isinstance(a, Branch) for a in args_t[:self.n_args]):
            if len(args_t) == self.n_args:  # Access one Keldysh block
                return self.components[tuple(a.value for a in args_t)]
            else:  # Pass extra indices to the block
                g = self.components[
                    tuple(a.value for a in args_t[:self.n_args])
                ]
                return g[args_t[self.n_args:]]

        # Access a single point of the time mesh
        elif len(args_t) == self.n_args and \
                all(isinstance(a, ContourPoint) for a in args_t):
            g = self.components[tuple(a.branch.value for a in args_t)]
            return g[tuple(a.t for a in args_t)
                     + (slice(None),) * len(self.non_time_mesh.components)]

        else:
            raise IndexError(f"Unrecognized index format: {args}")

    def __setitem__(self, args, value):
        args_t = args if isinstance(args, tuple) else (args,)

        assert len(args_t) >= self.n_args, \
            f"At least {self.n_args} arguments are required"

        if all(isinstance(a, Branch) for a in args_t[:self.n_args]):
            if len(args_t) == self.n_args:  # Access one Keldysh block
                self.components[tuple(a.value for a in args_t)] = value
            else:  # Pass extra indices to the block
                g = self.components[
                    tuple(a.value for a in args_t[:self.n_args])
                ]
                g[args_t[self.n_args:]] = value

        # Access a single point of the time mesh
        elif len(args_t) == self.n_args and \
                all(isinstance(a, ContourPoint) for a in args_t):
            g = self.components[tuple(a.branch.value for a in args_t)]
            g[tuple(a.t for a in args_t)
              + (slice(None),) * len(self.non_time_mesh.components)] = value

        else:
            raise IndexError(f"Unrecognized index format: {args}")

    #
    # Simple arithmetic
    #

    def __eq__(self, other):
        return self.mesh == other.mesh and \
            self.arg_index_shapes == other.arg_index_shapes and \
            (self.components == other.components).all()

    def __iadd__(self, other):
        assert self.mesh == other.mesh
        assert self.arg_index_shapes == other.arg_index_shapes
        self.components += other.components
        return self

    def __isub__(self, other):
        assert self.mesh == other.mesh
        assert self.arg_index_shapes == other.arg_index_shapes
        self.components -= other.components
        return self

    def __imul__(self, x):
        self.components *= x
        return self

    def __add__(self, other):
        res = deepcopy(self)
        res += other
        return res

    def __sub__(self, other):
        res = deepcopy(self)
        res -= other
        return res

    def __mul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __rmul__(self, x):
        res = deepcopy(self)
        res *= x
        return res

    def __neg__(self):
        res = deepcopy(self)
        res *= -1
        return res

    def __matmul__(self, other):
        r"""
        Contour convolution over the second argument of 'self' and
        the first argument of 'other'.
        """

        # General case
        if not (self.n_args == 2 and other.n_args == 2):
            return conv(self, other, [(-1, 0)])

        # High-accuracy approach for a convolution of two 2-point GFs

        self_l = lesser(self)
        self_g = greater(self)
        self_ret = retarded_ext(self)

        other_l = lesser(other)
        other_g = greater(other)
        other_adv = advanced_ext(other)

        conv_l = conv_ret_lg(self_ret, other_l) + conv_lg_adv(self_l, other_adv)
        conv_g = conv_ret_lg(self_ret, other_g) + conv_lg_adv(self_g, other_adv)

        return from_lesser_greater(
            conv_l,
            conv_g,
            n_left_target_axes=len(self.arg_index_shapes[0])
        )
    

class MixedKeldyshGF():
    pass



class KeldyshGFDetailed(KeldyshGF):
    def __init__( 
            self, *,
            mesh: Union[MeshReTime, MeshProduct],
            target_shape: Optional[Tuple[int, ...]] = None,
            arg_index_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
            #bosons: int = 0,
            fermions: int = 0
            ):
        self.fermions = fermions
         
        # The given number of point are fermions, the rest are bosons, fermions come before bosons
        super().__init__(mesh=mesh, target_shape = target_shape, arg_index_shapes = arg_index_shapes)
        self.bosons = self.n_args - self.fermions 
    #
    # Functions specific to the 2-fermion point GFs
    #

    def greater(self):
        r"""Returns the greater component of a 2- fermion point Keldysh Green's function"""
        assert self.fermions == 2, "Greater works only for 2-fermion point functions"  
        return self.components[Branch.BACKWARD.value, Branch.FORWARD.value, ...]
    
    def lesser(self):
        r"""Returns the lesser component of a 2- fermion point Keldysh Green's function"""
        assert self.fermions == 2, "Lesser works only for 2-fermion point functions"  
        return self.components[Branch.FORWARD.value, Branch.BACKWARD.value, ...]
    
    def retarded_ext(self):
        return self.greater()-self.lesser()
    
    def advanced_ext(self):
        return -self.retarded_ext()
        

    def __matmul__(self, other):
        pass
        





def target_dot(g: KeldyshGF,
               x: np.ndarray,
               g_arg: int,
               x_axes: Sequence[int]) -> KeldyshGF:
    r"""
    Given a Keldysh Green's function, compute a tensor contraction of its
    target axes corresponding to a specified argument with a NumPy array.

    g:      Keldysh Green's function.
    x:      NumPy array to contract with.
    g_arg:  `g`'s argument to be selected.
    x_axes: Axes of `x` used in the contraction. Dimensions of these axes must
            match the selected target subshape of `g`.
    """
    assert g_arg < g.n_args, \
        f"Invalid argument {g_arg} for the {g.n_args}-point GF"

    assert g.arg_index_shapes[g_arg] == tuple(x.shape[a] for a in x_axes), \
        "Axis dimensions of g's target space and x disagree"

    res_arg_index_shapes = list(g.arg_index_shapes)
    res_arg_index_shapes[g_arg] = tuple(s for a, s in enumerate(x.shape)
                                        if a not in x_axes)

    first_g_tg_axis = len(g.mesh.components) \
        + sum(len(ss) for ss in g.arg_index_shapes[:g_arg])
    g_tg_axis_range = range(first_g_tg_axis,
                            first_g_tg_axis + len(g.arg_index_shapes[g_arg]))
    n_free_x_axes = x.ndim - len(x_axes)

    res = KeldyshGF(mesh=g.mesh, arg_index_shapes=res_arg_index_shapes)
    for br in product(Branch, repeat=g.n_args):
        res[br].data[:] = np.moveaxis(
            np.tensordot(g[br].data, x, (g_tg_axis_range, x_axes)),
            range(-n_free_x_axes, 0),
            range(first_g_tg_axis, first_g_tg_axis + n_free_x_axes)
        )

    return res


#
# Functions specific to the 2-point GFs
#


def greater(g: KeldyshGF) -> Gf:
    r"""Returns the greater component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    return g[Branch.BACKWARD, Branch.FORWARD]


def lesser(g: KeldyshGF) -> Gf:
    r"""Returns the lesser component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    return g[Branch.FORWARD, Branch.BACKWARD]


def retarded(g: KeldyshGF) -> Gf:
    r"""Returns the retarded component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_ret = Gf(mesh=g.mesh, target_shape=g.target_shape)
    tril_idx = np.tril_indices(len(g.time_mesh.components[0]),
                               0,
                               len(g.time_mesh.components[1]))
    g_ret.data[tril_idx] = g_g.data[tril_idx] - g_l.data[tril_idx]
    return g_ret


def retarded_ext(g: KeldyshGF) -> Gf:
    r"""
    Returns the retarded component of a 2-point Keldysh Green's function
    G^r(t, t') continuously extended to the domain t < t'.
    The extended retarded component is defined as

        \tiled G^r(t, t') = G^>(t, t') - G^<(t, t').
    """
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_ret = Gf(mesh=g.mesh, target_shape=g.target_shape)
    g_ret.data[...] = g_g.data - g_l.data
    return g_ret


def advanced(g: KeldyshGF) -> Gf:
    r"""Returns the advanced component of a 2-point Keldysh Green's function"""
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_adv = Gf(mesh=g.mesh, target_shape=g.target_shape)
    triu_idx = np.triu_indices(len(g.time_mesh.components[0]),
                               0,
                               len(g.time_mesh.components[1]))
    g_adv.data[triu_idx] = g_l.data[triu_idx] - g_g.data[triu_idx]
    return g_adv


def advanced_ext(g: KeldyshGF) -> Gf:
    r"""
    Returns the advanced component of a 2-point Keldysh Green's function
    G^a(t, t') continuously extended to the domain t > t'.
    The extended advanced component is defined as

        \tiled G^a(t, t') = G^<(t, t') - G^>(t, t').
    """
    assert g.n_args == 2, "g must be a 2-point Green's function"
    g_g = g[Branch.BACKWARD, Branch.FORWARD]
    g_l = g[Branch.FORWARD, Branch.BACKWARD]
    g_adv = Gf(mesh=g.mesh, target_shape=g.target_shape)
    g_adv.data[...] = g_l.data - g_g.data
    return g_adv


def herm_conj(g: KeldyshGF) -> KeldyshGF:
    r"""
    Returns the Hermitian conjugate of a 2-point Keldysh Green's function as
    defined in the NESSi paper.
    """
    assert g.n_args == 2, "g must be a 2-point Green's function"

    g_g = greater(g)
    g_l = lesser(g)

    mesh_conj = MeshProduct(g.time_mesh[1],
                            g.time_mesh[0],
                            *g.non_time_mesh.components)
    target_shape_conj = g.arg_index_shapes[1] + g.arg_index_shapes[0]

    g_conj_g = Gf(mesh=mesh_conj, target_shape=target_shape_conj)
    g_conj_l = Gf(mesh=mesh_conj, target_shape=target_shape_conj)

    nli, nri = len(g.arg_index_shapes[0]), len(g.arg_index_shapes[1])

    axes_from = (0, 1, *range(-1, - nli - 1, -1))
    axes_to = (1, 0, *range(-1 - nli, -2 * nli - 1, -1))
    g_conj_g.data[:] = -np.conj(np.moveaxis(g_g.data, axes_from, axes_to))
    g_conj_l.data[:] = -np.conj(np.moveaxis(g_l.data, axes_from, axes_to))

    return from_lesser_greater(g_conj_l, g_conj_g, n_left_target_axes=nri)


def is_hermitian(g: KeldyshGF, *, atol=.0) -> bool:
    r"""
    Checks if a 2-point Keldysh Green's function is Hermitian in the sense of
    the NESSi paper.
    """
    assert g.n_args == 2, "g must be a 2-point Green's function"

    if g.time_mesh.components[0] != g.time_mesh.components[1]:
        return False
    if g.arg_index_shapes[0] != g.arg_index_shapes[1]:
        return False

    nli = len(g.arg_index_shapes[0])

    axes_from = (0, 1, *range(-1, - nli - 1, -1))
    axes_to = (1, 0, *range(-1 - nli, -2 * nli - 1, -1))

    for comp in (greater(g), lesser(g)):
        if not np.allclose(comp.data,
                           -np.conj(np.moveaxis(comp.data, axes_from, axes_to)),
                           atol=atol):
            return False

    return True


def from_lesser_greater(g_l: Gf, g_g: Gf, n_left_target_axes=None) -> KeldyshGF:
    r"""
    Construct a 2-point KeldyshGF object from a pair of lesser and greater
    real time Green's functions.
    """
    assert g_l.mesh == g_g.mesh
    assert g_l.target_shape == g_g.target_shape
    assert len(g_l.mesh.components) >= 2
    assert isinstance(g_l.mesh.components[0], MeshReTime) and \
           isinstance(g_l.mesh.components[1], MeshReTime)

    if n_left_target_axes is None:
        assert len(g_l.target_shape) % 2 == 0
        n_left_target_axes = len(g_l.target_shape) // 2

    arg_index_shapes = (g_l.target_shape[:n_left_target_axes],
                        g_l.target_shape[n_left_target_axes:])

    g = KeldyshGF(mesh=g_l.mesh, arg_index_shapes=arg_index_shapes)

    #
    # Fill Keldysh components
    #

    def ordered(z0, z1):
        return contour_ordering(z0, z1) == (0, 1)

    non_t_slice = (slice(None),) * len(g.non_time_mesh.components)

    # Aoki RMP, Eqs. (17)
    g[Branch.BACKWARD, Branch.FORWARD] = g_g
    g[Branch.FORWARD, Branch.BACKWARD] = g_l
    # Aoki RMP, Eqs. (15)
    for t0, t1 in g.time_mesh:
        sl = (t0, t1) + non_t_slice
        z0 = ContourPoint(Branch.FORWARD, t0)
        z1 = ContourPoint(Branch.FORWARD, t1)
        g[z0, z1] = g_g[sl] if ordered(z0, z1) else g_l[sl]
        z0 = ContourPoint(Branch.BACKWARD, t0)
        z1 = ContourPoint(Branch.BACKWARD, t1)
        g[z0, z1] = g_g[sl] if ordered(z0, z1) else g_l[sl]

    return g

#
# Functions specific to the 3-point GFs
#


def from_vertex3_pieces(G: Dict[Tuple[int, int, int], Gf]) -> KeldyshGF:
    r"""
    Construct a 3-point vertex from 6 real-time correlators.

    Each element of dictionary G corresponds to one permutation of operators
    in the correlator,
    $$
        G_{ijk}(t_0, t_1, t_2) = -\xi_{ijk} <O_i(t_i) O_j(t_j) O_k(t_k)>,
    $$
    where $O_0(t_0) = c(t_0)$, $O_1(t_1) = c^\dagger(t_1)$,
    $O_2(t_2) = \rho(t_2)$. $\xi_{ijk} = -1$ if permutation (ijk) swaps
    indices 0 and 1, and +1 otherwise.

    Keys are 3! = 6 triplets (i, j, k), which are permutations of (0, 1, 2)
    indicating the respective order of $c$, $c^\dagger$ and $\rho$.
    """
    assert len(G) == 6

    G0 = next(iter(G.values()))
    assert all(p.mesh == G0.mesh for p in G.values())
    assert all(p.target_shape == G0.target_shape for p in G.values())

    ts_len = len(G0.target_shape)
    assert ts_len % 3 == 0, \
        "Target shape of the pieces must contain a multiple of 3 elements"
    arg_index_shapes = (
        G0.target_shape[:ts_len // 3],
        G0.target_shape[ts_len // 3: 2 * ts_len // 3],
        G0.target_shape[2 * ts_len // 3:]
    )

    Lambda = KeldyshGF(mesh=G0.mesh, arg_index_shapes=arg_index_shapes)

    #
    # Fill Keldysh components
    #

    for a0, a1, a2 in product(Branch, repeat=3):
        for t0, t1, t2 in Lambda.mesh:
            z0 = ContourPoint(a0, t0)
            z1 = ContourPoint(a1, t1)
            z2 = ContourPoint(a2, t2)
            order = contour_ordering(z0, z1, z2)
            Lambda[z0, z1, z2] = G[order][t0, t1, t2]

    return Lambda


def conv(a: KeldyshGF,  # noqa: C901
         b: KeldyshGF,
         coupled_args: Sequence[Tuple[int, int]] = [],
         *,
         free_args: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
         ) -> KeldyshGF:
    r"""
    Compute a contour convolution and a sum over its corresponding target
    indices of two contour function 'a' and 'b' w.r.t. one or more pairs
    of arguments.

    a: First function in the convolution.
    b: Second function in the convolution.
    coupled_args: Each element of this tuple is a pair of indices of the
                  arguments to integrate over (one index for 'a' and the other
                  for 'b'). Negative indices are interpreted as counting
                  from the end of the respective argument list.
    free_args: Specifies how arguments of the convolution result are distributed
               between 'free' (non-integrated) arguments of 'a' and 'b'.
               For example, coupled_args = (1, 2),
               free_args = ((0, 2, 3), (1, 4)) may results in the following
               convolution,

               f(z_0, z_1, z_2, z_3, z_4) = \inf_C dz'
                    a(z_0, z', z_2, z_3) b(z_1, z_4, z').
    """

    # Normalize the tuple of coupled arguments by resolving the negative
    # indices, removing duplicate pairs and sorting the resulting list
    conv_args = []
    for arg_a, arg_b in coupled_args:
        assert -a.n_args <= arg_a < a.n_args, \
            f"Wrong argument number {arg_a} for the first function with " \
            "{a.n_args} arguments"
        assert -b.n_args <= arg_b < b.n_args, \
            f"Wrong argument number {arg_b} for the second function with " \
            "{b.n_args} arguments"
        conv_args.append((arg_a if arg_a >= 0 else a.n_args + arg_a,
                          arg_b if arg_b >= 0 else b.n_args + arg_b))

    conv_args = sorted(list(set(conv_args)))
    n_conv_args = len(conv_args)

    n_args_res = a.n_args + b.n_args - 2 * n_conv_args
    assert n_args_res >= 0

    #
    # Process free_args
    #

    if free_args is None:
        # By default, arguments of the results are substituted into a and b
        # in the original left-to-right order.
        free_args_a = list(range(a.n_args - n_conv_args))
        free_args_b = list(range(a.n_args - n_conv_args, n_args_res))
    else:
        assert len(free_args) == 2, "Expected 2 elements in free_args"
        free_args_a, free_args_b = free_args
        assert len(free_args_a) == a.n_args - n_conv_args, \
            f"There must be exactly {a.n_args - n_conv_args} in " \
            f"free_args[0], got {len(free_args_a)}"
        assert len(free_args_b) == b.n_args - n_conv_args, \
            f"There must be exactly {b.n_args - n_conv_args} in " \
            f"free_args[1], got {len(free_args_b)}"

    free_args_all = set(free_args_a).union(free_args_b)
    assert len(free_args_all) == len(free_args_a) + len(free_args_b), \
        "No repeated numbers are allowed in free_args"
    assert free_args_all == set(range(n_args_res)), \
        f"Numbers in free_args must fully cover the {range(n_args_res)}"

    #
    # Label arguments with numbers
    #

    arg_indices_a = [None] * a.n_args
    arg_indices_b = [None] * b.n_args

    # Process coupled arguments
    for i, (arg_a, arg_b) in enumerate(conv_args):
        arg_indices_a[arg_a] = n_args_res + i
        arg_indices_b[arg_b] = n_args_res + i
    # Process free arguments of a
    i = 0
    for arg_a, ind_a in enumerate(arg_indices_a):
        if ind_a is None:
            arg_indices_a[arg_a] = free_args_a[i]
            i += 1
    # Process free arguments of b
    i = 0
    for arg_b, ind_b in enumerate(arg_indices_b):
        if ind_b is None:
            arg_indices_b[arg_b] = free_args_b[i]
            i += 1

    #
    # Handle the time components of the meshes
    #

    min_t_mesh_size = a.integrator.order + 1

    # Check mesh compatibility and compute integration weights
    w = []
    for arg_a, arg_b in conv_args:
        t_mesh_a = a.time_mesh.components[arg_a]
        t_mesh_b = b.time_mesh.components[arg_b]
        assert t_mesh_a == t_mesh_b, \
            f"Incompatible time mesh components {t_mesh_a} and {t_mesh_b} for" \
            f" the coupled argument pair ({arg_a}, {arg_b})"
        assert len(t_mesh_a) >= min_t_mesh_size, \
            f"Time mesh of a's argument {arg_a} must have " \
            f"at least {min_t_mesh_size} nodes"

        w.append(a.integrator.weights_conv(t_mesh_a))

    # Gather components of the resulting mesh
    mesh_comps_res = [None] * n_args_res
    for arg_a, arg_res in enumerate(arg_indices_a):
        if arg_res < n_args_res:
            mesh_comps_res[arg_res] = a.time_mesh.components[arg_a]
    for arg_b, arg_res in enumerate(arg_indices_b):
        if arg_res < n_args_res:
            mesh_comps_res[arg_res] = b.time_mesh.components[arg_b]

    # Generate einsum() subscripts
    ts = subscripts['time']
    subs_a_t = ''.join([ts[i] for i in arg_indices_a])
    subs_b_t = ''.join([ts[i] for i in arg_indices_b])
    subs_res_t = ts[:n_args_res]
    subs_w = [ts[n_args_res + i] for i in range(len(conv_args))]

    #
    # Handle the non-time components of the meshes
    #

    nt_mesh_a = a.non_time_mesh.components
    nt_mesh_b = b.non_time_mesh.components

    nts = subscripts['nontime']
    if nt_mesh_a == nt_mesh_b:
        # If the non-time components of the meshes of a and b agree, then we
        # use the same non-time mesh for the result
        ss = nts[:len(nt_mesh_a)]
        subs_a_nt = ss
        subs_b_nt = ss
        subs_res_nt = ss

        mesh_comps_res += nt_mesh_a
    else:
        # Otherwise the result is defined on a direct product of the meshes.
        ss = nts[:len(nt_mesh_a)]
        subs_a_nt = ss
        subs_res_nt = ss
        ss = nts[len(nt_mesh_a):len(nt_mesh_a) + len(nt_mesh_b)]
        subs_b_nt = ss
        subs_res_nt += ss

        mesh_comps_res += nt_mesh_a + nt_mesh_b

    #
    # Handle the targets
    #

    # Check subshapes compatibility
    for arg_a, arg_b in conv_args:
        subshape_a = a.arg_index_shapes[arg_a]
        subshape_b = b.arg_index_shapes[arg_b]
        assert subshape_a == subshape_b, \
            f"Incompatible target sub-shapes {subshape_a} and {subshape_b} for"\
            f" the coupled argument pair ({arg_a}, {arg_b})"

    # Gather components of the resulting subshapes
    subshapes_res = [None] * n_args_res
    for arg_a, arg_res in enumerate(arg_indices_a):
        if arg_res < n_args_res:
            subshapes_res[arg_res] = a.arg_index_shapes[arg_a]
    for arg_b, arg_res in enumerate(arg_indices_b):
        if arg_res < n_args_res:
            subshapes_res[arg_res] = b.arg_index_shapes[arg_b]

    # Collect sub-shapes of all n_args_res + n_conv_args arguments
    subshapes_all = subshapes_res[:]
    for arg_a, _ in conv_args:
        subshapes_all.append(a.arg_index_shapes[arg_a])

    # Compile a partitioned list of target subscripts
    tgs_it = iter(subscripts['target'])
    tgs = [''.join(islice(tgs_it, len(subshape))) for subshape in subshapes_all]

    # Generate einsum() subscripts
    subs_a_tg = ''.join([tgs[i] for i in arg_indices_a])
    subs_b_tg = ''.join([tgs[i] for i in arg_indices_b])
    subs_res_tg = ''.join(tgs[:n_args_res])

    #
    # Perform summation
    #

    subs_a = subs_a_t + subs_a_nt + subs_a_tg
    subs_b = subs_b_t + subs_b_nt + subs_b_tg
    subs_res = subs_res_t + subs_res_nt + subs_res_tg

    subs = f"{subs_a}," + ','.join(subs_w) + f",{subs_b}->{subs_res}"

    #print(subs)

    res = KeldyshGF(mesh=MeshProduct(*mesh_comps_res),
                    arg_index_shapes=subshapes_res)

    for br in product(Branch, repeat=n_args_res + n_conv_args):
        br_a = tuple(br[i] for i in arg_indices_a)
        br_b = tuple(br[i] for i in arg_indices_b)
        br_res = tuple(br[:n_args_res])
        sign = (-1) ** br[n_args_res:].count(Branch.BACKWARD)
        res[br_res].data[:] += sign * np.einsum(subs,
                                                a[br_a].data,
                                                *w,
                                                b[br_b].data,
                                                optimize="optimal")

    return res
