"""Data-driven reduced basis without structure: POD of training solutions, then Galerkin.

Same reduced solve as ours, but the basis is not compatible (it need not contain gradients), so
it is the control for the value of the exact-sequence structure.
"""
import numpy as np
import torch

from compat_galerkin import lift

from .base import Method


class PODGalerkin(Method):
    def __init__(self, rank=32):
        self.rank = rank
        self.name = f"pod_galerkin_{rank}"

    def fit(self, family, train, device=None, **kwargs):
        cav = family.base
        snaps = train.e_ref[:, cav.interior_edges]
        S = np.hstack([snaps.real.T, snaps.imag.T])  # real basis: K and M are real
        U, s, _ = np.linalg.svd(S, full_matrices=False)
        r = min(self.rank, int((s > 1e-12 * s[0]).sum()))
        self.Q = torch.as_tensor(U[:, :r], dtype=lift.DTYPE)
        self.K = lift.to_torch_sparse(cav.interior(cav.K))
        return {"dim": r, "sv_tail": float(s[r] / s[0]) if r < len(s) else 0.0}

    def predict(self, family, inst, cav):
        M = lift.to_torch_sparse(cav.interior(cav.M))
        f = torch.as_tensor(inst.f[cav.interior_edges])
        e = np.zeros(cav.n_edges, dtype=complex)
        e[cav.interior_edges] = lift.reduced_solve(self.Q, self.K, M, f, inst.omega2).numpy()
        return e
