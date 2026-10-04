"""Gate 1: optimise a free nodal partition of unity on a single instance (no encoder).

Measures what the model class can represent: the lowest Galerkin error any W with n bumps
reaches on this instance, found by multi-start gradient descent on per-node logits.
"""
from dataclasses import dataclass

import numpy as np
import torch

from . import lift


@dataclass
class OracleProblem:
    """Torch tensors of one instance, interior edges only except A, G (all edges)."""

    A: torch.Tensor
    G: torch.Tensor
    idx: torch.Tensor
    K: torch.Tensor
    M: torch.Tensor
    f: torch.Tensor
    omega2: complex
    e_ref: torch.Tensor  # complex, interior edges
    boundary_mask: torch.Tensor
    coords: np.ndarray  # (n_nodes, 2)
    interior_nodes: np.ndarray

    @classmethod
    def from_instance(cls, cav, inst, e_ref):
        d = lift.DTYPE
        idx = cav.interior_edges
        return cls(
            A=lift.to_torch_sparse(cav.A),
            G=lift.to_torch_sparse(cav.G),
            idx=torch.as_tensor(idx),
            K=lift.to_torch_sparse(cav.interior(cav.K)),
            M=lift.to_torch_sparse(cav.interior(cav.M)),
            f=torch.as_tensor(inst.f[idx], dtype=d),
            omega2=inst.omega2,
            e_ref=torch.as_tensor(e_ref[idx]),
            boundary_mask=torch.as_tensor(cav.boundary_node_mask),
            coords=cav.mesh.p.T.copy(),
            interior_nodes=np.flatnonzero(~cav.boundary_node_mask),
        )

    def mnorm(self, x):
        """M-norm of a complex vector (M is real)."""
        mm = lambda v: torch.sparse.mm(self.M, v[:, None])[:, 0]
        return (x.real @ mm(x.real) + x.imag @ mm(x.imag)).sqrt()


def lifted_basis(p, logits):
    W = lift.make_pou(logits, p.boundary_mask)
    V = lift.whitney_lift(W, p.A, p.G)[p.idx]
    return lift.compress_chol(V, p.M)


def galerkin_error(p, logits):
    """Relative M-norm error of the Galerkin solution in the lifted space. Differentiable."""
    Q = lifted_basis(p, logits)
    e = lift.reduced_solve(Q, p.K, p.M, p.f, p.omega2)
    return p.mnorm(e - p.e_ref) / p.mnorm(p.e_ref)


def projection_error(p, logits):
    """Best-approximation error of e_ref in the same space (M-orthogonal projection)."""
    Q = lifted_basis(p, logits)
    MQ = torch.sparse.mm(p.M, Q)
    G = Q.T @ MQ
    c = lambda x: torch.linalg.solve(G, MQ.T @ x)
    proj = torch.complex(Q @ c(p.e_ref.real), Q @ c(p.e_ref.imag))
    return p.mnorm(proj - p.e_ref) / p.mnorm(p.e_ref)


def energy_error(p, logits):
    Q = lifted_basis(p, logits)
    e = lift.reduced_solve(Q, p.K, p.M, p.f, p.omega2)
    num, ref = p.mnorm(e) ** 2, p.mnorm(p.e_ref) ** 2
    return (num - ref).abs() / ref


def init_logits(p, n, kind, seed):
    rng = np.random.default_rng(seed)
    x = p.coords
    if kind == "rbf":
        pts = x[p.interior_nodes]
        c = pts[rng.choice(len(pts), n, replace=False)]
        for _ in range(10):  # Lloyd iterations
            lab = np.argmin(((pts[:, None] - c[None]) ** 2).sum(-1), axis=1)
            c = np.stack([pts[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(n)])
        d2 = ((x[:, None] - c[None]) ** 2).sum(-1)
        s2 = np.mean(np.sort(d2, axis=1)[:, 1]) + 1e-12  # typical squared distance to second-nearest centre
        logits = -d2 / (2 * s2) + 0.1 * rng.standard_normal((len(x), n))
    elif kind == "smooth":
        feats = np.sin(x @ rng.normal(0, 3.0, (2, 4 * n)) + rng.uniform(0, 2 * np.pi, 4 * n))
        logits = feats @ rng.normal(0, 1, (4 * n, n))
    else:
        raise ValueError(kind)
    return torch.as_tensor(logits, dtype=lift.DTYPE)


def optimise(p, n, kind, seed, steps=500, lr=0.05):
    """Adam on per-node logits. Returns dict with the final logits, history and convergence flag."""
    logits = init_logits(p, n, kind, seed).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    hist = []
    for _ in range(steps):
        loss = galerkin_error(p, logits)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    final = galerkin_error(p, logits.detach())
    tail = hist[int(0.8 * steps)]
    return {
        "logits": logits.detach(),
        "history": hist,
        "final": float(final),
        "converged": (tail - float(final)) / max(tail, 1e-12) <= 0.05,
    }
