"""Non-learned compatible space: the same Whitney lift with W = hat functions of a coarser mesh.

This is lowest-order Nedelec on the coarse mesh, expressed in our own pipeline. It is the direct
control for a learned W at equal reduced dimension: does learning beat plain geometry?
"""
import numpy as np
import torch
from skfem import Basis, ElementTriP1

from compat_galerkin import fem, lift

from .base import Method


class CoarseWhitney(Method):
    def __init__(self, coarse_refine=2):
        self.coarse_refine = coarse_refine
        self.name = f"coarse_whitney_r{coarse_refine}"
        self.Q = None

    def fit(self, family, train, device=None, **kwargs):
        cav = family.base
        coarse = fem.unit_square(self.coarse_refine)
        P = Basis(coarse, ElementTriP1()).probes(cav.mesh.p).toarray()  # fine nodes x coarse nodes
        W = torch.as_tensor(P, dtype=lift.DTYPE)
        cb = np.zeros(coarse.p.shape[1], dtype=bool)
        cb[np.unique(coarse.facets[:, coarse.boundary_facets()])] = True
        lo, hi = coarse.facets
        keep = ~(cb[lo] & cb[hi])  # PEC: drop coarse edges lying on the boundary
        pairs = torch.as_tensor(np.vstack([lo[keep], hi[keep]]), dtype=torch.long)
        A, G = lift.to_torch_sparse(cav.A), lift.to_torch_sparse(cav.G)
        idx = torch.as_tensor(cav.interior_edges)
        V = lift.whitney_lift(W, A, G, pairs)[idx]
        self.Q = lift.compress(V, lift.to_torch_sparse(cav.interior(cav.M)))
        self.K = lift.to_torch_sparse(cav.interior(cav.K))
        return {"dim": self.Q.shape[1]}

    def predict(self, family, inst, cav):
        M = lift.to_torch_sparse(cav.interior(cav.M))
        f = torch.as_tensor(inst.f[cav.interior_edges])
        e = np.zeros(cav.n_edges, dtype=complex)
        e[cav.interior_edges] = lift.reduced_solve(self.Q, self.K, M, f, inst.omega2).numpy()
        return e
