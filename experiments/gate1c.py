"""Gate 1c: a spectrum-matching term added to the Gate 1b `mixed` loss.

Protocol and decision rules are in the README ("Gate 1c protocol"). Usage:
    uv run python -m experiments.gate1c [--steps 500] [--sizes 4,8,12,16] [--weights 0.1,1,10]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

from compat_galerkin import fem, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.diag_band_spectrum import BAND, ZERO, reduced_eigs
from experiments.gate1 import build_instance, frequencies
from experiments.gate1b import BUDGET, CONV_TOL, HELD, QOI_TOL, STARTS, TAN, TRAIN, Band, band_errors, evaluate

K_TARGETS = 4
EPS_M = 1e-6  # ridge on the reduced mass matrix inside the differentiable eigenvalue computation


def spectral_term(p, Q, lam):
    """Mean squared log-ratio between the lowest reduced physical eigenvalues and the true ones."""
    Mr = Q.T @ torch.sparse.mm(p.M, Q)
    Kr = Q.T @ torch.sparse.mm(p.K, Q)
    Mr = 0.5 * (Mr + Mr.T) + EPS_M * torch.eye(Mr.shape[0], dtype=Mr.dtype)
    L = torch.linalg.cholesky(Mr)
    Y = torch.linalg.solve_triangular(L, Kr, upper=False)
    Kt = torch.linalg.solve_triangular(L, Y.T, upper=False).T
    ev = torch.linalg.eigvalsh(0.5 * (Kt + Kt.T))
    mu = ev[ev > ZERO][: len(lam)]
    return ((mu.log() - lam[: len(mu)].log()) ** 2).mean()


def objective(tr, p, Q, lam, w):
    gal, proj, _ = band_errors(tr, Q)
    return gal.mean() + proj.mean() + w * spectral_term(p, Q, lam)


def train(tr, p, lam, n, w, kind, seed, steps, lr=0.05):
    lg = oracle.init_logits(p, n, kind, seed).requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    hist = []
    for _ in range(steps):
        loss = objective(tr, p, oracle.lifted_basis(p, lg), lam, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    lg = lg.detach()
    with torch.no_grad():
        final = float(objective(tr, p, oracle.lifted_basis(p, lg), lam, w))
    tail = hist[int(0.8 * steps)]
    return {"logits": lg, "final": final, "converged": (tail - final) / max(tail, 1e-12) <= CONV_TOL}


def spectrum_status(p, Q, lam1):
    eig = reduced_eigs(p, Q)
    phys = eig[eig > ZERO]
    hw = TAN * lam1
    near = phys[abs(phys - lam1) < hw]
    in_band = phys[(phys >= BAND[0]) & (phys <= BAND[1])]
    ghosts = in_band[abs(in_band - lam1) >= hw]
    return {"n_at_true_mode": len(near), "n_ghosts_in_band": len(ghosts),
            "spectrum_fixed": len(near) >= 2 and len(ghosts) == 0, "in_band": str(np.round(in_band, 3).tolist())}


def verdict(rows):
    aff = [r for r in rows if r["dim"] <= BUDGET]
    passed = [r for r in aff if r["held_energy_max"] < QOI_TOL]
    if passed:
        b = min(passed, key=lambda r: r["held_energy_max"])
        return (f"PASS: {len(passed)} of {len(aff)} affordable configurations; best w={b['w']}, n={b['n']}, "
                f"held-out energy {b['held_energy_max']:.3f}")
    fixed = [r for r in aff if r["spectrum_fixed"]]
    if fixed:
        b = min(fixed, key=lambda r: r["held_energy_max"])
        tag = "SPECTRUM FIXED BUT INACCURATE" if b["converged"] else "INCONCLUSIVE (unconverged)"
        return f"{tag}: {len(fixed)} configurations with the spectrum fixed; best w={b['w']}, n={b['n']}, held-out energy {b['held_energy_max']:.3f}"
    return "SPECTRUM NOT REACHED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="4,8,12,16")
    ap.add_argument("--weights", default="0.1,1,10")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    weights = [float(x) for x in a.weights.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    w_all = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray(), eigvals_only=True)
    lam = torch.as_tensor(w_all[w_all > ZERO][:K_TARGETS], dtype=torch.float64)
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    print(f"target eigenvalues {np.round(lam.numpy(), 3)}, half-width {TAN * lam1:.3f}, budget dim <= {BUDGET}", flush=True)

    rows = []
    for w in weights:
        print(f"\nw = {w}")
        for n in sizes:
            t0 = time.time()
            runs = [train(tr, p, lam, n, w, kind, seed, a.steps) for kind, seed in STARTS]
            b = min(runs, key=lambda r: r["final"])
            with torch.no_grad():
                Q = oracle.lifted_basis(p, b["logits"])
                ev = evaluate(tr, he, Q)
            sp = spectrum_status(p, Q, lam1)
            row = {"w": w, "n": n, "dim": Q.shape[1], "converged": b["converged"], **ev, **sp}
            rows.append(row)
            print(f"  n={n:2d} dim={row['dim']:4d} held: gal max {ev['held_gal_max']:.3f} proj max {ev['held_proj_max']:.3f} "
                  f"energy max {ev['held_energy_max']:.3f} | train gal mean {ev['train_gal_mean']:.3f} | "
                  f"modes at true: {sp['n_at_true_mode']}, ghosts in band: {sp['n_ghosts_in_band']}, band {sp['in_band']} "
                  f"| fixed={sp['spectrum_fixed']} conv={b['converged']} ({time.time() - t0:.0f}s)", flush=True)

    v = verdict(rows)
    print(f"\nVerdict: {v}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1c_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
