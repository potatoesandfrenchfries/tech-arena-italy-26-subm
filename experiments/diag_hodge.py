"""Post hoc diagnostic for Gate 1e (outside the protocol).

Gate 1e's registered verdict was NO IMPROVEMENT, but the spectrum was not fixed in any configuration,
so it is unclear whether the Hodge split was tested in the regime where the Gate 1c floor appears.
Two checks:

  1. Training breakdown: retrain the Gate 1e n = 12 space (same seeds) and print, for each start, the
     Galerkin, projection and spectral terms and the lowest reduced transverse eigenvalues.
  2. Post hoc Hodge split: take the Gate 1c space that did fix the spectrum (w = 10, n = 16), project
     the curl-free content out of its lifted vectors, solve the transverse part in that space, and add
     the exact curl-free part. No retraining. Reports the error and the new spectrum.

    uv run python -m experiments.diag_hodge [--steps 500]
"""
import argparse

import numpy as np
import scipy.linalg as sl
import torch

from compat_galerkin import fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments import gate1c
from experiments.diag_band_spectrum import BAND, ZERO, reduced_eigs
from experiments.gate1 import build_instance, frequencies
from experiments.gate1b import HELD, STARTS, TAN, TRAIN, Band
from experiments.gate1c import K_TARGETS, spectral_term
from experiments.gate1e import JITTER, Hodge, basis, evaluate, hodge_errors, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    w_all = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray(), eigvals_only=True)
    lam = torch.as_tensor(w_all[w_all > ZERO][:K_TARGETS], dtype=torch.float64)
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    hodge = Hodge(cav, f, p)
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    print(f"true lowest eigenvalues {np.round(lam.numpy(), 3)}, half-width {TAN * lam1:.3f}")

    print("\n1. Gate 1e training breakdown, n = 12, w = 10")
    for kind, seed in STARTS:
        r = train(tr, hodge, p, lam, 12, 10.0, kind, seed, a.steps)
        with torch.no_grad():
            Q = basis(p, hodge, r["logits"])
            gal, proj, _ = hodge_errors(tr, hodge, p, Q)
            spec = float(spectral_term(p, Q, lam))
        eig = reduced_eigs(p, Q)
        eig = eig[eig > ZERO]
        print(f"  start {kind}-{seed}: final objective {r['final']:.3f} = gal {float(gal.mean()):.3f} + proj "
              f"{float(proj.mean()):.3f} + 10 x spec {spec:.4f}; lowest reduced eigenvalues {np.round(eig[:6], 3).tolist()}")

    print("\n2. Post hoc Hodge split of the Gate 1c space (w = 10, n = 16)")
    runs = [gate1c.train(tr, p, lam, 16, 10.0, kind, seed, a.steps) for kind, seed in STARTS]
    b = min(runs, key=lambda r: r["final"])
    with torch.no_grad():
        Q_before = oracle.lifted_basis(p, b["logits"])
        W = lift.make_pou(b["logits"], p.boundary_mask)
        V = lift.whitney_lift(W, p.A, p.G)[p.idx]
        Q_after = lift.compress_chol(hodge.project_out(V, p.M), p.M, jitter=JITTER)
        ev = evaluate(tr, he, hodge, p, Q_after)
    eig_b, eig_a = reduced_eigs(p, Q_before), reduced_eigs(p, Q_after)
    eig_b, eig_a = eig_b[eig_b > ZERO], eig_a[eig_a > ZERO]
    inb = lambda e: np.round(e[(e >= BAND[0]) & (e <= BAND[1])], 3).tolist()
    print(f"  before (Gate 1c space, no split): in-band eigenvalues {inb(eig_b)}")
    print(f"  after the split: in-band eigenvalues {inb(eig_a)}, lowest 4 {np.round(eig_a[:4], 3).tolist()}")
    print(f"  after the split: held-out gal mean {ev['held_gal_mean']:.3f} max {ev['held_gal_max']:.3f} | proj max "
          f"{ev['held_proj_max']:.3f} | energy max {ev['held_energy_max']:.3f}")


if __name__ == "__main__":
    main()
