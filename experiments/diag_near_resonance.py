"""Post hoc diagnostics for the Gate 1 near-resonance case (outside the registered protocol).

Gate 1 left `near` inconclusive: the Galerkin-optimised W gives error ~1, yet the best-approximation
error of spaces of the same size is small. This script asks why. For each size n:

  1. Optimise W for the projection (best-approximation) error instead of the Galerkin error.
  2. Solve in that space with Galerkin, and with a minimal-residual (least-squares) solve.
  3. Continue optimising the Galerkin error from the projection-optimised W.
  4. Inspect the reduced spectrum against the true spectrum (spurious low eigenvalues?).
  5. Control: add the exact fine-mesh resonant eigenvectors to the space. If the Galerkin error then
     falls to the projection floor, the failure is the reduced spectrum, not the field.

The same quantities are reported for the non-learned coarse Nedelec spaces.

    uv run python -m experiments.diag_near_resonance [--steps 500] [--sizes 8,12,16]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spla
import torch

from baselines import CoarseWhitney
from compat_galerkin import fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.gate1 import STARTS, build_instance, frequencies

ZERO = 1e-6  # reduced eigenvalues below this are the curl-free kernel


def rel_err(p, e):
    return float(p.mnorm(e - p.e_ref) / p.mnorm(p.e_ref))


def projection_err(p, Q):
    """Best-approximation error of e_ref in span(Q). Gram solve, so Q need not be orthonormal."""
    MQ = torch.sparse.mm(p.M, Q)
    G = Q.T @ MQ
    c = lambda x: torch.linalg.lstsq(G, MQ.T @ x).solution
    return rel_err(p, torch.complex(Q @ c(p.e_ref.real), Q @ c(p.e_ref.imag)))


def galerkin(p, Q):
    return lift.reduced_solve(Q, p.K, p.M, p.f, p.omega2)


class MinimalResidual:
    """argmin_y || f - (K - omega^2 M) Q y ||_{M^-1}, the dual norm natural to this problem."""

    def __init__(self, cav, p):
        self.Kn, self.Mn = cav.interior(cav.K).tocsc(), cav.interior(cav.M).tocsc()
        self.lu = spla.splu(self.Mn)
        self.w2, self.f = p.omega2, p.f.numpy().astype(complex)

    def _minv(self, X):
        return self.lu.solve(np.ascontiguousarray(X.real)) + 1j * self.lu.solve(np.ascontiguousarray(X.imag))

    def __call__(self, Q):
        Qn = Q.numpy()
        B = self.Kn @ Qn - self.w2 * (self.Mn @ Qn)
        H = B.conj().T @ self._minv(B)
        y = np.linalg.lstsq(H, B.conj().T @ self._minv(self.f), rcond=None)[0]
        return torch.as_tensor(Qn @ y)


def reduced_spectrum(p, Q):
    Kr = (Q.T @ torch.sparse.mm(p.K, Q)).numpy()
    Mr = (Q.T @ torch.sparse.mm(p.M, Q)).numpy()
    return sl.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T), eigvals_only=True)


def spectrum_summary(ev, lam1, w2_re, width):
    phys = ev[ev > ZERO]
    return {
        "kernel_dim": int((ev <= ZERO).sum()),
        "n_below_lam1": int((phys < lam1 * (1 - 1e-3)).sum()),
        "nearest_to_omega2": float(phys[np.argmin(abs(phys - w2_re))]),
        "within_width": bool(np.min(abs(phys - w2_re)) < width),
    }


def _adam(loss_fn, logits, steps, lr=0.05):
    lg = logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    for _ in range(steps):
        loss = loss_fn(lg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lg.detach()


def optimise_projection(p, n, steps):
    """Best of the four Gate 1 starts, by projection error."""
    best = None
    for kind, seed in STARTS:
        lg = _adam(lambda x: oracle.projection_error(p, x), oracle.init_logits(p, n, kind, seed), steps)
        with torch.no_grad():
            val = float(oracle.projection_error(p, lg))
        if best is None or val < best[0]:
            best = (val, lg)
    return best[1]


def true_resonant_vectors(cav, lam1):
    """M-orthonormal fine-mesh eigenvectors with eigenvalue lam1 (the degenerate pair)."""
    w, v = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray())
    return torch.as_tensor(v[:, abs(w - lam1) < 1e-3 * lam1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="8,12,16")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    freqs, modes = frequencies(cav)
    lam1, w2 = modes[0], freqs["near"]
    inst = Instance(eps, w2 * (1 + 0.05j), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst, fem.solve_fine(cav, inst.omega2, f))
    mr = MinimalResidual(cav, p)
    width = 0.05 * w2  # half the imaginary part of omega^2
    X = true_resonant_vectors(cav, lam1)
    true_ev = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray(), eigvals_only=True)
    print(f"near: omega^2 = {w2:.3f}, true lambda_1 = {lam1:.3f} (x{X.shape[1]}), half-width = {width:.3f}")
    print("true spectrum, first physical:", np.round(true_ev[true_ev > ZERO][:4], 3))

    rows = []

    def record(label, n, Q):
        ev = reduced_spectrum(p, Q)
        Qa = lift.compress(torch.cat([Q, X], dim=1), p.M)
        row = {
            "space": label, "n": n, "dim": Q.shape[1],
            "proj_err": projection_err(p, Q),
            "galerkin_err": rel_err(p, galerkin(p, Q)),
            "minres_err": rel_err(p, mr(Q)),
            "augmented_galerkin_err": rel_err(p, galerkin(p, Qa)),
            "augmented_minres_err": rel_err(p, mr(Qa)),
            **spectrum_summary(ev, lam1, w2, width),
            "lowest_three_phys": str([round(float(x), 3) for x in ev[ev > ZERO][:3]]),
        }
        rows.append(row)
        print(f"  {label:<20} dim={row['dim']:4d} proj={row['proj_err']:.3f} galerkin={row['galerkin_err']:.3f} "
              f"minres={row['minres_err']:.3f} | +exact modes: galerkin={row['augmented_galerkin_err']:.3f} "
              f"minres={row['augmented_minres_err']:.3f} | "
              f"below lam1: {row['n_below_lam1']}, lowest {row['lowest_three_phys']}, "
              f"nearest omega^2 {row['nearest_to_omega2']:.3f} (within width: {row['within_width']})", flush=True)

    for n in sizes:
        t0 = time.time()
        print(f"\nn = {n}")
        lg = optimise_projection(p, n, a.steps)
        with torch.no_grad():
            record("proj-optimised W", n, oracle.lifted_basis(p, lg))
        lg = _adam(lambda x: oracle.galerkin_error(p, x), lg, a.steps)
        with torch.no_grad():
            record("+ Galerkin steps", n, oracle.lifted_basis(p, lg))
        print(f"  ({time.time() - t0:.0f}s)")

    print("\nControls: non-learned compatible spaces")
    for r in (1, 2, 3):
        m = CoarseWhitney(r)
        m.fit(fam, None)
        record(f"coarse_r{r}", 0, m.Q)

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"diag_near_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
