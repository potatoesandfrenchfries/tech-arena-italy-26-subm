"""One learned partition of unity shared by every instance (no encoder).

The Whitney lift of the partition is the same for all geometries; only the whitening and the reduced
matrices change with the instance (through the epsilon-weighted mass matrix). Trained on a set of
instances by minimising the Galerkin error against the fine solutions, with mini-batches. This is the
fixed, geometry-independent compatible basis: the control that tells whether an encoder (a W that
depends on the geometry) is needed at all.
"""
import numpy as np
import torch

from compat_galerkin import lift, oracle

from .base import Method


class SharedWhitney(Method):
    def __init__(self, n=8, steps=500, batch=16, lr=0.05, starts=(("rbf", 0), ("smooth", 0))):
        self.n, self.steps, self.batch, self.lr, self.starts = n, steps, batch, lr, starts
        self.name = f"shared_whitney_n{n}"
        self.V = None

    @staticmethod
    def _error(p, V):
        Q = lift.compress_chol(V, p.M)
        e = lift.reduced_solve(Q, p.K, p.M, p.f, p.omega2)
        return p.mnorm(e - p.e_ref) / p.mnorm(p.e_ref)

    @staticmethod
    def _lift(p0, logits):
        W = lift.make_pou(logits, p0.boundary_mask)
        return lift.whitney_lift(W, p0.A, p0.G)[p0.idx]

    def fit(self, family, train, device=None, **kwargs):
        probs = [oracle.OracleProblem.from_instance(family.cavity(inst), inst, e_ref)
                 for inst, e_ref in zip(train.instances, train.e_ref)]
        p0 = probs[0]
        best = None
        for kind, seed in self.starts:
            logits = oracle.init_logits(p0, self.n, kind, seed).requires_grad_(True)
            opt = torch.optim.Adam([logits], lr=self.lr)
            rng = np.random.default_rng(seed)
            for _ in range(self.steps):
                V = self._lift(p0, logits)
                sel = rng.choice(len(probs), min(self.batch, len(probs)), replace=False)
                loss = torch.stack([self._error(probs[i], V) for i in sel]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                V = self._lift(p0, logits.detach())
                final = float(torch.stack([self._error(p, V) for p in probs]).mean())
            if best is None or final < best[0]:
                best = (final, logits.detach())
        self.train_error, self.logits = best
        self.V = self._lift(p0, self.logits)
        s = torch.linalg.svdvals(self.V)
        return {"dim": int((s > 1e-8 * s[0]).sum()), "columns": self.V.shape[1], "train_error": self.train_error}

    def predict(self, family, inst, cav):
        K, M = lift.to_torch_sparse(cav.interior(cav.K)), lift.to_torch_sparse(cav.interior(cav.M))
        f = torch.as_tensor(inst.f[cav.interior_edges])
        Q = lift.compress_chol(self.V, M)
        e = np.zeros(cav.n_edges, dtype=complex)
        e[cav.interior_edges] = lift.reduced_solve(Q, K, M, f, inst.omega2).numpy()
        return e
