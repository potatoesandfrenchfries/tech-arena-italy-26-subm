"""Gate 1f: Hodge split plus an energy-norm capture loss for the true resonant modes.

Second attempt on the Hodge-split idea after Gate 1e. Protocol and decision rules are in the README
("Gate 1f protocol"). Usage:
    uv run python -m experiments.gate1f [--steps 500] [--sizes 4,8,12,16] [--weights 0.1,1]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

from compat_galerkin import fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.diag_band_spectrum import ZERO
from experiments.gate1 import build_instance, frequencies
from experiments.gate1b import BUDGET, CONV_TOL, HELD, QOI_TOL, STARTS, TAN, TRAIN, Band
from experiments.gate1c import K_TARGETS, spectral_term, spectrum_status
from experiments.gate1e import RIDGE, Hodge, basis, evaluate, hodge_errors

W_SPEC = 10.0
LOG_DELTA = 1e-6
IMPROVE_TOL = 0.15
GROUPS = [[0, 1], [2], [3]]  # resonant pair, lambda_3, lambda_4


def capture_err2(p, Q, MPhi, lamK):
    """Relative energy-norm (K + M) best-approximation error^2 of each true eigenvector in span(Q)."""
    Mr = Q.T @ torch.sparse.mm(p.M, Q)
    Kr = Q.T @ torch.sparse.mm(p.K, Q)
    S = Kr + Mr + RIDGE * torch.eye(Q.shape[1], dtype=Q.dtype)
    B = Q.T @ MPhi
    Z = torch.linalg.solve(S, B)
    return (1 - (1 + lamK) * (B * Z).sum(0)).clamp(min=0)


def capture_term(err2):
    return torch.stack([torch.log(err2[g].mean() + LOG_DELTA) for g in GROUPS]).mean()


def objective(tr, hodge, p, Q, lam, MPhi, lamK, w_cap):
    gal, proj, _ = hodge_errors(tr, hodge, p, Q)
    return gal.mean() + proj.mean() + W_SPEC * spectral_term(p, Q, lam) + w_cap * capture_term(capture_err2(p, Q, MPhi, lamK))


def train(tr, hodge, p, lam, MPhi, lamK, n, w_cap, kind, seed, steps, lr=0.05):
    lg = oracle.init_logits(p, n, kind, seed).requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    hist = []
    for _ in range(steps):
        loss = objective(tr, hodge, p, basis(p, hodge, lg), lam, MPhi, lamK, w_cap)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    lg = lg.detach()
    with torch.no_grad():
        final = float(objective(tr, hodge, p, basis(p, hodge, lg), lam, MPhi, lamK, w_cap))
    tail = hist[int(0.8 * steps)]
    return {"logits": lg, "final": final, "converged": (tail - final) / max(abs(tail), 1e-12) <= CONV_TOL}


def verdict(rows):
    aff = [r for r in rows if r["dim"] <= BUDGET]
    passed = [r for r in aff if r["held_energy_max"] < QOI_TOL]
    if passed:
        b = min(passed, key=lambda r: r["held_energy_max"])
        return (f"PASS: {len(passed)} of {len(aff)} affordable configurations; best w_cap={b['w_cap']}, n={b['n']}, "
                f"held-out energy {b['held_energy_max']:.3f}")
    improved = [r for r in aff if r["held_gal_mean"] < IMPROVE_TOL]
    if improved:
        b = min(improved, key=lambda r: r["held_gal_mean"])
        tag = "IMPROVED BUT NOT PASSING" if b["converged"] else "INCONCLUSIVE (unconverged)"
        return (f"{tag}: {len(improved)} of {len(aff)} affordable configurations below {IMPROVE_TOL}; best w_cap={b['w_cap']}, "
                f"n={b['n']}, held-out gal mean {b['held_gal_mean']:.3f}, energy {b['held_energy_max']:.3f}")
    b = min(aff, key=lambda r: r["held_gal_mean"])
    tag = "NO IMPROVEMENT" if b["converged"] else "INCONCLUSIVE (unconverged)"
    return f"{tag}: best affordable held-out gal mean {b['held_gal_mean']:.3f} (w_cap={b['w_cap']}, n={b['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="4,8,12,16")
    ap.add_argument("--weights", default="0.1,1")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    weights = [float(x) for x in a.weights.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    w_all, X = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray())
    lam = torch.as_tensor(w_all[w_all > ZERO][:K_TARGETS], dtype=torch.float64)
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    hodge = Hodge(cav, f, p)
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    Phi = torch.as_tensor(X[:, w_all > ZERO][:, :K_TARGETS])
    MPhi = torch.sparse.mm(p.M, Phi)
    lamK = lam
    print(f"w_spec = {W_SPEC}, w_cap in {weights}, target eigenvalues {np.round(lam.numpy(), 3)}, budget dim <= {BUDGET}")

    # Sanity checks of the capture term (reported, not part of the verdict).
    with torch.no_grad():
        exact = capture_err2(p, torch.as_tensor(X[:, w_all > ZERO][:, :10]), MPhi, lamK)
        g = torch.Generator().manual_seed(0)
        Qr = torch.linalg.qr(hodge.project_out(torch.randn(len(cav.interior_edges), 16, generator=g, dtype=lift.DTYPE), p.M))[0]
        rand = capture_err2(p, Qr, MPhi, lamK)
    print(f"capture err^2 with the 10 lowest exact transverse eigenvectors: {np.round(exact.numpy(), 6).tolist()} (expected ~0)")
    print(f"capture err^2 with a random transverse 16-dimensional space: {np.round(rand.numpy(), 3).tolist()} (expected near 1)", flush=True)

    rows = []
    for w_cap in weights:
        print(f"\nw_cap = {w_cap}")
        for n in sizes:
            t0 = time.time()
            runs = [train(tr, hodge, p, lam, MPhi, lamK, n, w_cap, kind, seed, a.steps) for kind, seed in STARTS]
            b = min(runs, key=lambda r: r["final"])
            with torch.no_grad():
                Q = basis(p, hodge, b["logits"])
                ev = evaluate(tr, he, hodge, p, Q)
                err2 = capture_err2(p, Q, MPhi, lamK)
                W = lift.make_pou(b["logits"], p.boundary_mask)
                V = hodge.project_out(lift.whitney_lift(W, p.A, p.G, torch.triu_indices(n, n, offset=1))[p.idx], p.M)
                s = torch.linalg.svdvals(V)
                dim = int((s > 1e-8 * s[0]).sum())
            sp = spectrum_status(p, Q, lam1)
            cap = [float(err2[g].mean()) for g in GROUPS]
            row = {"w_cap": w_cap, "n": n, "dim": dim, "converged": b["converged"], "cap_pair": cap[0],
                   "cap_lam3": cap[1], "cap_lam4": cap[2], **ev, **sp}
            rows.append(row)
            print(f"  n={n:2d} dim={dim:4d} held: gal mean {ev['held_gal_mean']:.3f} max {ev['held_gal_max']:.3f} "
                  f"proj max {ev['held_proj_max']:.3f} energy max {ev['held_energy_max']:.3f} | capture err^2 pair "
                  f"{cap[0]:.4f} l3 {cap[1]:.4f} l4 {cap[2]:.4f} | modes at true: {sp['n_at_true_mode']}, ghosts: "
                  f"{sp['n_ghosts_in_band']}, band {sp['in_band']} | fixed={sp['spectrum_fixed']} "
                  f"conv={b['converged']} ({time.time() - t0:.0f}s)", flush=True)

    v = verdict(rows)
    print(f"\nVerdict: {v}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1f_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
