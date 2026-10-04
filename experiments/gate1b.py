"""Gate 1b, stage (a): one W shared across a frequency band containing a resonance.

Protocol and decision rules are in the README ("Gate 1b protocol"). Usage:
    uv run python -m experiments.gate1b [--steps 500] [--sizes 4,8,12,16,24]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

from baselines import CoarseWhitney
from compat_galerkin import fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.gate1 import build_instance, numerical_rank

TAN = 0.05
TRAIN = 6.5 + 0.3125 * np.arange(9)
HELD = 0.5 * (TRAIN[:-1] + TRAIN[1:])
STARTS = [("rbf", 0), ("smooth", 0)]
LOSSES = ("galerkin", "proj", "mixed")
BUDGET, QOI_TOL, PLATEAU, CONV_TOL = 150, 0.05, 0.8, 0.05
C128 = torch.complex128


class Band:
    """A set of frequencies on one instance, with their fine-mesh reference solutions."""

    def __init__(self, p, cav, f, w2s):
        self.p = p
        self.w = torch.as_tensor(np.asarray(w2s) * (1 + 1j * TAN), dtype=C128)
        refs = [fem.solve_fine(cav, w * (1 + 1j * TAN), f)[cav.interior_edges] for w in w2s]
        self.R = torch.as_tensor(np.stack(refs, axis=1))  # (interior edges, F)

    def mnorm(self, X):
        mm = lambda v: torch.sparse.mm(self.p.M, v)
        return ((X.real * mm(X.real)).sum(0) + (X.imag * mm(X.imag)).sum(0)).sqrt()


def band_errors(band, Q):
    """Per-frequency relative M-norm errors for the space span(Q): Galerkin, projection, energy.

    Differentiable in Q. Q need not be exactly M-orthonormal."""
    p, R = band.p, band.R
    MQ = torch.sparse.mm(p.M, Q)
    Mr = Q.T @ MQ
    Kr = Q.T @ torch.sparse.mm(p.K, Q)
    Qc = Q.to(C128)
    A = Kr.to(C128)[None] - band.w[:, None, None] * Mr.to(C128)[None]
    rhs = (Qc.T @ p.f.to(C128))[None, :, None].expand(len(band.w), -1, 1)
    E = Qc @ torch.linalg.solve(A, rhs)[:, :, 0].T
    ref = band.mnorm(R)
    c = lambda X: torch.linalg.solve(Mr, MQ.T @ X)
    P = torch.complex(Q @ c(R.real), Q @ c(R.imag))
    return band.mnorm(E - R) / ref, band.mnorm(P - R) / ref, (band.mnorm(E) ** 2 - ref**2).abs() / ref**2


def objective(name, band, Q):
    gal, proj, _ = band_errors(band, Q)
    return {"galerkin": gal.mean(), "proj": proj.mean(), "mixed": gal.mean() + proj.mean()}[name]


def train(tr, n, loss_name, kind, seed, steps, lr=0.05):
    p = tr.p
    lg = oracle.init_logits(p, n, kind, seed).requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    hist = []
    for _ in range(steps):
        loss = objective(loss_name, tr, oracle.lifted_basis(p, lg))
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    lg = lg.detach()
    with torch.no_grad():
        final = float(objective(loss_name, tr, oracle.lifted_basis(p, lg)))
    tail = hist[int(0.8 * steps)]
    return {"logits": lg, "final": final, "converged": (tail - final) / max(tail, 1e-12) <= CONV_TOL}


def evaluate(tr, he, Q):
    with torch.no_grad():
        g_t, p_t, e_t = band_errors(tr, Q)
        g_h, p_h, e_h = band_errors(he, Q)
    w = int(g_h.argmax())
    return {
        "train_gal_mean": float(g_t.mean()), "train_gal_max": float(g_t.max()),
        "train_proj_mean": float(p_t.mean()), "train_energy_max": float(e_t.max()),
        "held_gal_mean": float(g_h.mean()), "held_gal_max": float(g_h.max()),
        "held_proj_mean": float(p_h.mean()), "held_proj_max": float(p_h.max()),
        "held_energy_max": float(e_h.max()), "held_ratio_worst": float(g_h[w] / p_h[w]),
        "held_worst_omega2": float(HELD[w]),
    }


def modal_space(cav, fam, p, k):
    """Lowest k exact eigenvectors plus gradients of the level-3 coarse hats (curl-free part)."""
    w, v = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray())
    modes = v[:, w > 1e-6][:, :k]
    grads = (cav.G.tocsr()[cav.interior_edges] @ fam.coarse_hats(3)).toarray()
    V = torch.as_tensor(np.hstack([modes, grads]), dtype=lift.DTYPE)
    return lift.compress(V, p.M)


def verdict(rows):
    ora = [r for r in rows if r["kind"] == "oracle" and r["dim"] <= BUDGET]
    passed = [r for r in ora if r["held_energy_max"] < QOI_TOL]
    if passed:
        b = min(passed, key=lambda r: r["dim"])
        return f"PASS (loss={b['loss']}, n={b['n']}, dim={b['dim']})"
    interp = [r for r in ora if r["train_energy_max"] < QOI_TOL]
    if interp:
        b = min(interp, key=lambda r: r["held_energy_max"])
        return ("INTERPOLATION FAIL" if b["converged"] else "INCONCLUSIVE (unconverged)") + \
            f" (loss={b['loss']}, n={b['n']}: train energy ok, held-out {b['held_energy_max']:.2f})"
    stab = [r for r in ora if r["held_proj_max"] < QOI_TOL]
    if stab:
        b = min(stab, key=lambda r: r["held_proj_max"])
        return ("STABILITY-LIMITED" if b["converged"] else "INCONCLUSIVE (unconverged)") + \
            f" (loss={b['loss']}, n={b['n']}: held-out proj {b['held_proj_max']:.3f}, galerkin {b['held_gal_max']:.2f})"
    by = {(r["loss"], r["n"]): r for r in rows if r["kind"] == "oracle"}
    ratios = [by[(l, 24)]["held_proj_max"] / by[(l, 16)]["held_proj_max"] for l in LOSSES if (l, 16) in by and (l, 24) in by]
    if ratios and min(ratios) >= PLATEAU and all(r["converged"] for r in ora):
        return "CAPACITY FAIL (plateaued)"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="4,8,12,16,24")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    print(f"train omega^2 {np.round(TRAIN, 3)}\nheld-out {np.round(HELD, 3)}\nbudget dim <= {BUDGET}", flush=True)

    cols = ["kind", "loss", "n", "dim", "converged"]
    rows = []

    def add(kind, loss, n, dim, conv, Q):
        row = {"kind": kind, "loss": loss, "n": n, "dim": dim, "converged": conv, **evaluate(tr, he, Q)}
        rows.append(row)
        print(f"  {kind:<12}{loss:<9}n={n:3d} dim={dim:4d} held: gal max {row['held_gal_max']:.3f} proj max "
              f"{row['held_proj_max']:.3f} energy max {row['held_energy_max']:.3f} | train gal mean "
              f"{row['train_gal_mean']:.3f} | worst omega^2 {row['held_worst_omega2']:.2f} conv={conv}", flush=True)

    print("\nComparators")
    for r in (1, 2, 3):
        m = CoarseWhitney(r)
        info = m.fit(fam, None)
        add("coarse", "-", r, info["dim"], True, m.Q)
    for k in (2, 10):
        Q = modal_space(cav, fam, p, k)
        add("modal+grad", "-", k, Q.shape[1], True, Q)

    for loss in LOSSES:
        print(f"\nLoss: {loss}")
        for n in sizes:
            t0 = time.time()
            runs = [train(tr, n, loss, kind, seed, a.steps) for kind, seed in STARTS]
            b = min(runs, key=lambda r: r["final"])  # chosen on the training objective only
            with torch.no_grad():
                Q = oracle.lifted_basis(p, b["logits"])
                dim = numerical_rank(p, b["logits"])
            add("oracle", loss, n, dim, b["converged"], Q)
            print(f"    ({time.time() - t0:.0f}s)", flush=True)

    v = verdict(rows)
    print(f"\nVerdict: {v}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1b_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + [k for k in rows[0] if k not in cols])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
