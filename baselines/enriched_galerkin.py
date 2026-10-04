"""The FNO's field added to a compatible Whitney space, then a Galerkin solve (variant C1, Study 6b).

The reduced space is span(Q0, Re e_fno, Im e_fno) with Q0 a non-learned coarse Whitney basis. Whitening uses the
instance's mass matrix. The answer then satisfies the Galerkin identities (power balance, PEC) by construction, and
the space is defined on any mesh because both the coarse Whitney basis and the FNO are.
"""
import numpy as np
import torch

from compat_galerkin import lift

from .base import Method


class EnrichedGalerkin(Method):
    def __init__(self, coarse, fno_predict):
        """coarse: a fitted `CoarseWhitney` (Q, K); fno_predict(family, inst, cav) -> complex edge vector."""
        self.coarse, self.fno_predict = coarse, fno_predict
        self.name = f"enriched_{coarse.name}"

    @torch.no_grad()
    def predict(self, family, inst, cav):
        idx = cav.interior_edges
        M = lift.to_torch_sparse(cav.interior(cav.M))
        f = torch.as_tensor(inst.f[idx])
        ef = np.asarray(self.fno_predict(family, inst, cav))[idx]
        cols = [self.coarse.Q]
        for part in (ef.real, ef.imag):
            v = torch.as_tensor(np.ascontiguousarray(part), dtype=lift.DTYPE)[:, None]
            nrm = float((v.T @ torch.sparse.mm(M, v)).sqrt())
            if nrm > 0:
                cols.append(v / nrm)
        Q = lift.compress_chol(torch.cat(cols, dim=1), M)
        e = np.zeros(cav.n_edges, dtype=complex)
        e[idx] = lift.reduced_solve(Q, self.coarse.K, M, f, inst.omega2).numpy()
        return e
