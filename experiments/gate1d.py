"""Gate 1d: the Gate 1c loss with a larger curl-free space inside the training loop.

Fixed exact gradients of coarse hat functions are appended to the lifted vectors before whitening, so
the Galerkin, projection and spectral terms are all computed on the enlarged space. Protocol and
decision rules are in the README ("Gate 1d protocol"). Usage:
    uv run python -m experiments.gate1d [--steps 500] [--sizes 4,8,12,16] [--levels 1,2] [--weight 10]
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
from experiments.gate1b import BUDGET, CONV_TOL, HELD, QOI_TOL, STARTS, TAN, TRAIN, Band, band_errors, evaluate
from experiments.gate1c import K_TARGETS, spectral_term, spectrum_status

IMPROVE_TOL = 0.15  # half of the Gate 1c floor (0.29)


def basis(p, logits, Gx):
    """Whitened lifted vectors plus fixed exact gradients Gx (edges x k)."""
    W = lift.make_pou(logits, p.boundary_mask)
    V = torch.cat([lift.whitney_lift(W, p.A, p.G)[p.idx], Gx], dim=1)
    return lift.compress_chol(V, p.M)


def objective(tr, p, Q, lam, w):
    gal, proj, _ = band_errors(tr, Q)
    return gal.mean() + proj.mean() + w * spectral_term(p, Q, lam)


def train(tr, p, Gx, lam, n, w, kind, seed, steps, lr=0.05):
    lg = oracle.init_logits(p, n, kind, seed).requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    hist = []
    for _ in range(steps):
        loss = objective(tr, p, basis(p, lg, Gx), lam, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    lg = lg.detach()
    with torch.no_grad():
        final = float(objective(tr, p, basis(p, lg, Gx), lam, w))
    tail = hist[int(0.8 * steps)]
    return {"logits": lg, "final": final, "converged": (tail - final) / max(tail, 1e-12) <= CONV_TOL}


def verdict(rows):
    aff = [r for r in rows if r["dim"] <= BUDGET]
    passed = [r for r in aff if r["held_energy_max"] < QOI_TOL]
    if passed:
        b = min(passed, key=lambda r: r["held_energy_max"])
        return (f"PASS: {len(passed)} of {len(aff)} affordable configurations; best level={b['level']}, n={b['n']}, "
                f"held-out energy {b['held_energy_max']:.3f}")
    improved = [r for r in aff if r["held_gal_mean"] < IMPROVE_TOL]
    if improved:
        b = min(improved, key=lambda r: r["held_gal_mean"])
        tag = "IMPROVED BUT NOT PASSING" if b["converged"] else "INCONCLUSIVE (unconverged)"
        return (f"{tag}: {len(improved)} of {len(aff)} affordable configurations below {IMPROVE_TOL}; best level={b['level']}, "
                f"n={b['n']}, held-out gal mean {b['held_gal_mean']:.3f}, energy {b['held_energy_max']:.3f}")
    b = min(aff, key=lambda r: r["held_gal_mean"])
    tag = "NO IMPROVEMENT" if b["converged"] else "INCONCLUSIVE (unconverged)"
    return f"{tag}: best affordable held-out gal mean {b['held_gal_mean']:.3f} (level={b['level']}, n={b['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="4,8,12,16")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--weight", type=float, default=10.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    levels = [int(x) for x in a.levels.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    w_all = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray(), eigvals_only=True)
    lam = torch.as_tensor(w_all[w_all > ZERO][:K_TARGETS], dtype=torch.float64)
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    Gi = cav.G.tocsr()[cav.interior_edges]
    print(f"w = {a.weight}, target eigenvalues {np.round(lam.numpy(), 3)}, budget dim <= {BUDGET}", flush=True)

    rows = []
    for level in levels:
        Gx = torch.as_tensor((Gi @ fam.coarse_hats(level)).toarray(), dtype=lift.DTYPE)
        print(f"\nlevel {level}: {Gx.shape[1]} added gradients")
        for n in sizes:
            t0 = time.time()
            runs = [train(tr, p, Gx, lam, n, a.weight, kind, seed, a.steps) for kind, seed in STARTS]
            b = min(runs, key=lambda r: r["final"])
            with torch.no_grad():
                Q = basis(p, b["logits"], Gx)
                ev = evaluate(tr, he, Q)
            sp = spectrum_status(p, Q, lam1)
            row = {"level": level, "added": Gx.shape[1], "n": n, "dim": Q.shape[1], "converged": b["converged"], **ev, **sp}
            rows.append(row)
            note = "" if row["dim"] <= BUDGET else " [over budget, excluded from verdict]"
            print(f"  n={n:2d} dim={row['dim']:4d}{note} held: gal mean {ev['held_gal_mean']:.3f} max {ev['held_gal_max']:.3f} "
                  f"proj max {ev['held_proj_max']:.3f} energy max {ev['held_energy_max']:.3f} | train gal mean "
                  f"{ev['train_gal_mean']:.3f} | modes at true: {sp['n_at_true_mode']}, ghosts in band: "
                  f"{sp['n_ghosts_in_band']}, band {sp['in_band']} | fixed={sp['spectrum_fixed']} "
                  f"conv={b['converged']} ({time.time() - t0:.0f}s)", flush=True)

    v = verdict(rows)
    print(f"\nVerdict: {v}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1d_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
