"""Gate 0: structural identities that must hold for ANY partition of unity, with no learning.

Each metric is a relative residual; see README "Main risks and how we decide".
"""
import numpy as np
import scipy.sparse.linalg as spla
import torch
from skfem import Basis, BilinearForm, ElementTriP1
from skfem.helpers import dot, grad

from . import fem, lift


def random_pou(cav, n=6, seed=0, scale=3.0):
    """Smooth random PoU: softmax of random Fourier features of the node coordinates."""
    rng = np.random.default_rng(seed)
    x = cav.mesh.p.T
    feats = np.sin(x @ rng.normal(0, scale, (2, 4 * n)) + rng.uniform(0, 2 * np.pi, 4 * n))
    logits = torch.as_tensor(feats @ rng.normal(0, 1, (4 * n, n)), dtype=lift.DTYPE)
    return lift.make_pou(logits, torch.as_tensor(cav.boundary_node_mask))


def dof_convention(cav, seed=0):
    """L2 projection of grad(u) onto N1 must equal G u: pins the sign/orientation of the dofs."""
    bp = Basis(cav.mesh, ElementTriP1(), quadrature=cav.basis.quadrature)
    mass = BilinearForm(lambda u, v, w: dot(u, v)).assemble(cav.basis)
    mixed = BilinearForm(lambda u, v, w: dot(grad(u), v)).assemble(bp, cav.basis)
    u = np.random.default_rng(seed).standard_normal(cav.n_nodes)
    c = spla.spsolve(mass.tocsc(), mixed @ u)
    gu = cav.G @ u
    return np.linalg.norm(c - gu) / np.linalg.norm(gu)


def curl_grad(cav, seed=0):
    u = np.random.default_rng(seed).standard_normal(cav.n_nodes)
    gu = cav.G @ u
    return np.linalg.norm(cav.K @ gu) / (spla.norm(cav.K) * np.linalg.norm(gu))


def gate0(cav, n=6, seed=0, omega2=4.0 + 0.2j):
    A, G = lift.to_torch_sparse(cav.A), lift.to_torch_sparse(cav.G)
    W = random_pou(cav, n, seed)
    m = W.shape[1]
    out = {"dof_convention": dof_convention(cav, seed), "curl_grad": curl_grad(cav, seed)}

    # sum_j c_ij = -grad W_i, on all edges
    AW, GW = (torch.sparse.mm(A, W), torch.sparse.mm(G, W))
    C = AW[:, :, None] * GW[:, None, :] - AW[:, None, :] * GW[:, :, None]
    out["lift_identity"] = float((GW + C.sum(dim=2)).abs().max() / GW.abs().max())

    V = lift.whitney_lift(W, A, G)
    out["pec_trace"] = float(V[torch.as_tensor(cav.boundary_edges)].abs().max())

    idx = torch.as_tensor(cav.interior_edges)
    K, M = (lift.to_torch_sparse(cav.interior(x)) for x in (cav.K, cav.M))
    Q = lift.compress(V[idx], M)
    out["dim_reduced"] = Q.shape[1]
    out["dim_whitney"] = V.shape[1]

    f = torch.as_tensor(fem.current_source(cav)[cav.interior_edges], dtype=lift.DTYPE)
    e = lift.reduced_solve(Q, K, M, f, omega2).numpy()
    Kn, Mn, fn = cav.interior(cav.K), cav.interior(cav.M), f.numpy()

    # weak Gauss law: v^T (K - w2 M) e = v^T f with K v = 0 gives  -w2 v^T M e = v^T f.
    # Tested with v = grad(psi) for psi in the reduced 0-form space, and (control) for random
    # fine-mesh interior functions, which are NOT in the reduced space and must violate it.
    def gauss(psi):
        v = (cav.G @ psi)[cav.interior_edges]
        lhs, rhs = -omega2 * (v @ (Mn @ e)), v @ fn
        return abs(lhs - rhs) / (abs(lhs) + abs(rhs))

    out["gauss_reduced"] = max(gauss(W[:, i].numpy()) for i in range(n))
    rng = np.random.default_rng(seed + 1)
    ctrl = [rng.standard_normal(cav.n_nodes) * ~cav.boundary_node_mask for _ in range(n)]
    out["gauss_control"] = min(gauss(p) for p in ctrl)

    # complex power balance: e^H (K - w2 M) e = e^H f
    lhs = np.vdot(e, Kn @ e - omega2 * (Mn @ e))
    rhs = np.vdot(e, fn)
    out["power_balance"] = abs(lhs - rhs) / abs(rhs)

    # informational only: error of an untrained random PoU against the fine solution
    ef = fem.solve_fine(cav, omega2, fem.current_source(cav))[cav.interior_edges]
    out["info_rel_err_random_W"] = np.linalg.norm(e - ef) / np.linalg.norm(ef)
    return out
