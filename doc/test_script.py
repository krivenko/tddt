import numpy as np
from itertools import product

from triqs.gf import MeshReTime  # TRIQS real time mesh
from triqs.gf import MeshProduct # A direct product of 1D meshes

from tddt.keldysh import KeldyshGF, KeldyshGFDetailed

t_max = 5.0
N_t = 11

t_mesh = MeshReTime(0.0, t_max, N_t)
ttt_mesh=MeshProduct(t_mesh, t_mesh, t_mesh)

a = KeldyshGFDetailed(mesh = ttt_mesh, arg_index_shapes=((2,),(2,),(1,)), fermions = 2)
#print(a.components)

print(a.retarded_ext())

t_points = list(t_mesh)
t1, t2 = t_points[0], t_points[1]

print(a.advanced_ext())