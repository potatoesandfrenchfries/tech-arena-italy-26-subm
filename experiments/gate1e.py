"""Gate 1e: Hodge split. The curl-free part of the solution is solved exactly on the fine mesh with one
coercive scalar solve per instance; the learned partition of unity only builds the transverse reduced space.

Protocol and decision rules are in the README ("Gate 1e protocol"). Usage:
    uv run python -m experiments.gate1e [--steps 500] [--sizes 4,8,12,16] [--weight 10]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spla
import torch

from compat_galerkin import fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.diag_band_spectrum import ZERO
from experiments.gate1 import build_instance, frequencies
from experiments.gate1b import BUDGET, CONV_TOL, HELD, QOI_TOL, STARTS, TAN, TRAIN, Band
from experiments.gate1c import K_TARGETS, spectral_term, spectrum_status

C128 = torch.complex128
JITTER = 1e-8  # relative jitter of the whitening (protocol)
RIDGE = 1e-8  # ridge on the reduced matrices, for the rank deficiency left by the whitening
IMPROVE_TOL = 0.15


class Hodge:
    """Exact curl-free part: phi solves (Gn^T M Gn) phi = Gn^T f; e_L = -(1/omega^2) Gn phi."""

    def __init__(self, cav, f, p):
        idx = cav.interior_edges
        nodes = np.flatnonzero(~cav.boundary_node_mask)
        Gn = cav.G.tocsr()[idx][:, nodes]
        Mn = cav.interior(cav.M).tocsc()
        A0 = (Gn.T @ Mn @ Gn).tocsc()
        self.Gn, self.GnT = lift.to_torch_sparse(Gn), lift.to_torch_sparse(Gn.T.tocsr())
        self.L0 = torch.linalg.cholesky(torch.as_tensor(A0.toarray()))
        t0 = time.perf_counter()
        phi = spla.spsolve(A0, Gn.T @ f[idx])
        self.solve_seconds = time.perf_counter() - t0
        self.gphi = torch.as_tensor(Gn @ phi)
        self.fT = torch.as_tensor(f[idx], dtype=lift.DTYPE) - torch.sparse.mm(p.M, self.gphi[:, None])[:, 0]

    def project_out(self, V, M):
        GtMV = torch.sparse.mm(self.GnT, torch.sparse.mm(M, V))
        return V - torch.sparse.mm(self.Gn, torch.cholesky_solve(GtMV, self.L0))


def basis(p, hodge, logits):
    """Whitened transverse reduced basis built from the pairs among the n learned bumps.

    Pairs involving the remainder function are exactly dependent on these once the curl-free content is
    projected out (sum_j w_ij = -grad W_i lies in the kernel), so they are not built."""
    n = logits.shape[1]
    W = lift.make_pou(logits, p.boundary_mask)
    V = lift.whitney_lift(W, p.A, p.G, torch.triu_indices(n, n, offset=1))[p.idx]
    return lift.compress_chol(hodge.project_out(V, p.M), p.M, jitter=JITTER)


def hodge_errors(band, hodge, p, Q):
    """Per-frequency Galerkin, transverse-projection and energy errors of e = e_L + Q y, relative to ||e_ref||."""
    EL = -hodge.gphi.to(C128)[:, None] / band.w[None, :]
    RT = band.R - EL
    MQ = torch.sparse.mm(p.M, Q)
    Mr = Q.T @ MQ
    Kr = Q.T @ torch.sparse.mm(p.K, Q)
    eye = torch.eye(Q.shape[1], dtype=Q.dtype)
    rhs = (Q.T @ hodge.fT).to(C128)
    A = (Kr + RIDGE * eye).to(C128)[None] - band.w[:, None, None] * Mr.to(C128)[None]
    Y = torch.linalg.solve(A, rhs[None, :, None].expand(len(band.w), -1, 1))[:, :, 0]
    ET = Q.to(C128) @ Y.T
    ref = band.mnorm(band.R)
    c = lambda X: torch.linalg.solve(Mr + RIDGE * eye, MQ.T @ X)
    P = torch.complex(Q @ c(RT.real), Q @ c(RT.imag))
    gal = band.mnorm(ET - RT) / ref
    proj = band.mnorm(P - RT) / ref
    energy = (band.mnorm(EL + ET) ** 2 - ref**2).abs() / ref**2
    return gal, proj, energy


def objective(tr, hodge, p, Q, lam, w):
    gal, proj, _ = hodge_errors(tr, hodge, p, Q)
    return gal.mean() + proj.mean() + w * spectral_term(p, Q, lam)


def train(tr, hodge, p, lam, n, w, kind, seed, steps, lr=0.05):
    lg = oracle.init_logits(p, n, kind, seed).requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    hist = []
    for _ in range(steps):
        loss = objective(tr, hodge, p, basis(p, hodge, lg), lam, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    lg = lg.detach()
    with torch.no_grad():
        final = float(objective(tr, hodge, p, basis(p, hodge, lg), lam, w))
    tail = hist[int(0.8 * steps)]
    return {"logits": lg, "final": final, "converged": (tail - final) / max(tail, 1e-12) <= CONV_TOL}


def evaluate(tr, he, hodge, p, Q):
    with torch.no_grad():
        g_t, p_t, e_t = hodge_errors(tr, hodge, p, Q)
        g_h, p_h, e_h = hodge_errors(he, hodge, p, Q)
    k = int(g_h.argmax())
    return {
        "train_gal_mean": float(g_t.mean()), "train_gal_max": float(g_t.max()),
        "train_proj_mean": float(p_t.mean()), "train_energy_max": float(e_t.max()),
        "held_gal_mean": float(g_h.mean()), "held_gal_max": float(g_h.max()),
        "held_proj_mean": float(p_h.mean()), "held_proj_max": float(p_h.max()),
        "held_energy_max": float(e_h.max()), "held_worst_omega2": float(HELD[k]),
    }


def verdict(rows):
    aff = [r for r in rows if r["dim"] <= BUDGET]
    passed = [r for r in aff if r["held_energy_max"] < QOI_TOL]
    if passed:
        b = min(passed, key=lambda r: r["held_energy_max"])
        return f"PASS: {len(passed)} of {len(aff)} affordable configurations; best n={b['n']}, held-out energy {b['held_energy_max']:.3f}"
    improved = [r for r in aff if r["held_gal_mean"] < IMPROVE_TOL]
    if improved:
        b = min(improved, key=lambda r: r["held_gal_mean"])
        tag = "IMPROVED BUT NOT PASSING" if b["converged"] else "INCONCLUSIVE (unconverged)"
        return (f"{tag}: {len(improved)} of {len(aff)} affordable configurations below {IMPROVE_TOL}; best n={b['n']}, "
                f"held-out gal mean {b['held_gal_mean']:.3f}, energy {b['held_energy_max']:.3f}")
    b = min(aff, key=lambda r: r["held_gal_mean"])
    tag = "NO IMPROVEMENT" if b["converged"] else "INCONCLUSIVE (unconverged)"
    return f"{tag}: best affordable held-out gal mean {b['held_gal_mean']:.3f} (n={b['n']})"


def sanity_checks(cav, f, p, hodge, tr, he, Mn):
    """Implementation checks reported before the run (not part of the verdict)."""
    EL = -hodge.gphi.to(C128)[:, None] / he.w[None, :]
    RT = he.R - EL
    resid = torch.sparse.mm(hodge.GnT, torch.sparse.mm(p.M, RT.real)).norm() + torch.sparse.mm(hodge.GnT, torch.sparse.mm(p.M, RT.imag)).norm()
    print(f"kernel-annihilation residual ||Gn^T M (e_ref - e_L)|| / ||e_ref|| = "
          f"{float(resid / he.R.abs().norm()):.2e}  (required < 1e-10)")
    w, X = sl.eigh(cav.interior(cav.K).toarray(), Mn.toarray())
    Xt = torch.as_tensor(X[:, w > ZERO][:, :10])
    with torch.no_grad():
        g, pr, e = hodge_errors(he, hodge, p, Xt)
    print(f"Hodge solve with the 10 lowest exact transverse eigenvectors: held-out gal max {float(g.max()):.4f}, "
          f"proj max {float(pr.max()):.4f}, energy max {float(e.max()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="4,8,12,16")
    ap.add_argument("--weight", type=float, default=10.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    Mn = cav.interior(cav.M).tocsc()
    w_all = sl.eigh(cav.interior(cav.K).toarray(), Mn.toarray(), eigvals_only=True)
    lam = torch.as_tensor(w_all[w_all > ZERO][:K_TARGETS], dtype=torch.float64)
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    t0 = time.perf_counter()
    e0 = fem.solve_fine(cav, inst0.omega2, f)
    fine_seconds = time.perf_counter() - t0
    p = oracle.OracleProblem.from_instance(cav, inst0, e0)
    hodge = Hodge(cav, f, p)
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    print(f"w = {a.weight}, target eigenvalues {np.round(lam.numpy(), 3)}, budget dim <= {BUDGET}")
    print(f"cost (informational, one instance): scalar curl-free solve {hodge.solve_seconds * 1e3:.1f} ms "
          f"({hodge.Gn.shape[1]} unknowns) vs fine Maxwell solve {fine_seconds * 1e3:.1f} ms ({len(cav.interior_edges)} unknowns)")
    sanity_checks(cav, f, p, hodge, tr, he, Mn)

    rows = []
    for n in sizes:
        t0 = time.time()
        runs = [train(tr, hodge, p, lam, n, a.weight, kind, seed, a.steps) for kind, seed in STARTS]
        b = min(runs, key=lambda r: r["final"])
        with torch.no_grad():
            Q = basis(p, hodge, b["logits"])
            ev = evaluate(tr, he, hodge, p, Q)
            W = lift.make_pou(b["logits"], p.boundary_mask)
            V = hodge.project_out(lift.whitney_lift(W, p.A, p.G, torch.triu_indices(n, n, offset=1))[p.idx], p.M)
            s = torch.linalg.svdvals(V)
            dim = int((s > 1e-8 * s[0]).sum())
        sp = spectrum_status(p, Q, lam1)
        row = {"n": n, "dim": dim, "columns": Q.shape[1], "converged": b["converged"], **ev, **sp}
        rows.append(row)
        print(f"  n={n:2d} dim={dim:4d} held: gal mean {ev['held_gal_mean']:.3f} max {ev['held_gal_max']:.3f} "
              f"proj max {ev['held_proj_max']:.3f} energy max {ev['held_energy_max']:.3f} | train gal mean "
              f"{ev['train_gal_mean']:.3f} | modes at true: {sp['n_at_true_mode']}, ghosts in band: "
              f"{sp['n_ghosts_in_band']}, band {sp['in_band']} | fixed={sp['spectrum_fixed']} "
              f"conv={b['converged']} ({time.time() - t0:.0f}s)", flush=True)

    v = verdict(rows)
    print(f"\nVerdict: {v}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1e_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
