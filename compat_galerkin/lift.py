"""Compatible reduced space from a partition of unity (PoU), and the reduced solve.

Given nodal functions W_i (a PoU), the Whitney 1-forms w_ij = W_i grad W_j - W_j grad W_i are
lifted to fine-mesh edge cochains c_ij(ab) = avg_ab(W_i) (grad W_j)_ab - avg_ab(W_j) (grad W_i)_ab.
Because sum_j W_j = 1, grad W_i = -sum_j w_ij, so the reduced 1-form space contains the gradients
of the reduced 0-form space. Everything here is differentiable in W.
"""
import numpy as np
import scipy.sparse as sp
import torch

DTYPE = torch.float64


def to_torch_sparse(mat, device=None):
    coo = sp.coo_matrix(mat)
    idx = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    t = torch.sparse_coo_tensor(idx, torch.as_tensor(coo.data, dtype=DTYPE), coo.shape).coalesce()
    return t if device is None else t.to(device)


def make_pou(logits, boundary_mask):
    """Nodal PoU with n bumps that vanish on the boundary plus a remainder function.

    logits: (n_nodes, n). Returns W (n_nodes, n + 1) with columns summing to 1; the last
    column equals 1 on the boundary and every other column equals 0 there (PEC compatible).
    """
    n_nodes, n = logits.shape
    ext = torch.cat([logits, torch.zeros(n_nodes, 1, dtype=logits.dtype, device=logits.device)], dim=1)
    bumps = torch.softmax(ext, dim=1)[:, :n] * (~boundary_mask).to(logits.dtype)[:, None]
    return torch.cat([bumps, 1.0 - bumps.sum(dim=1, keepdim=True)], dim=1)


def whitney_lift(W, A, G, pairs=None):
    """Edge cochains of Whitney 1-forms. A, G are torch sparse (edges x nodes).

    pairs: (2, P) index tensor of (i, j) to build; default is all m(m-1)/2 pairs. Pairs whose
    supports are disjoint give zero columns, so a sparse pair list (e.g. mesh edges) is enough.
    """
    AW = torch.sparse.mm(A, W)
    GW = torch.sparse.mm(G, W)
    m = W.shape[1]
    i, j = torch.triu_indices(m, m, offset=1, device=W.device) if pairs is None else pairs.to(W.device)
    return AW[:, i] * GW[:, j] - AW[:, j] * GW[:, i]


def compress(V, M, rtol=1e-10):
    """M-orthonormal basis of the column space of V (the w_ij are linearly dependent)."""
    S = V.T @ torch.sparse.mm(M, V)
    evals, U = torch.linalg.eigh(S)
    keep = evals > rtol * evals.max()
    return V @ (U[:, keep] / evals[keep].sqrt())


def reduced_solve(Q, K, M, f, omega2):
    """Galerkin solve in span(Q). K, M torch sparse; Q real (edges x q); f, omega2 may be complex.

    The reduced matrices are real, so a frequency sweep only repeats a q x q dense solve.
    """
    Kr = Q.T @ torch.sparse.mm(K, Q)
    Mr = Q.T @ torch.sparse.mm(M, Q)
    A = (Kr - omega2 * Mr).to(torch.complex128)
    Qc = Q.to(torch.complex128)
    y = torch.linalg.solve(A, Qc.T @ f.to(torch.complex128))
    return Qc @ y


def compress_chol(V, M, jitter=1e-10):
    """Differentiable M-orthonormalisation of V by Cholesky whitening with a relative jitter.

    Unlike `compress` (eigh), the backward pass is stable when the w_ij are nearly dependent:
    directions with singular value below ~sqrt(jitter) are damped instead of amplified.
    """
    S = V.T @ torch.sparse.mm(M, V)
    S = S + jitter * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    L = torch.linalg.cholesky(S)
    return torch.linalg.solve_triangular(L, V.T, upper=False).T
