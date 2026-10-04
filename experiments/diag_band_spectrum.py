"""Post hoc diagnostic for Gate 1b: where are the reduced eigenvalues relative to the band?

Gate 1b was stability-limited. Two mechanisms are possible for a space whose projection error is
small but whose Galerkin error is large:
  missing  - no reduced eigenvalue within the resonance half-width of the true mode (7.604), so the
             resonance response is not reproduced;
  spurious - reduced eigenvalues inside the band away from the true mode, which create ghost
             resonances at other frequencies.
This retrains the Gate 1b spaces with the registered protocol (same seeds, steps and selection), prints
their errors as a reproduction check against the README rows, then reports the reduced spectrum in
the band and, per frequency, the distance to the nearest reduced eigenvalue in half-widths.

    uv run python -m experiments.diag_band_spectrum [--steps 500] [--sizes 8,16,24]
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
from experiments.gate1 import build_instance, frequencies
from experiments.gate1b import HELD, STARTS, TAN, TRAIN, Band, band_errors, evaluate, train

ZERO = 1e-6  # reduced eigenvalues below this are the curl-free kernel
BAND = (6.5, 9.0)


def reduced_eigs(p, Q):
    """Eigenvalues of the reduced curl-curl problem, on a well-conditioned M-orthonormal basis."""
    Mr = (Q.T @ torch.sparse.mm(p.M, Q)).numpy()
    Kr = (Q.T @ torch.sparse.mm(p.K, Q)).numpy()
    s, U = np.linalg.eigh(0.5 * (Mr + Mr.T))
    keep = s > 1e-8 * s.max()
    T = U[:, keep] / np.sqrt(s[keep])
    return np.linalg.eigvalsh(T.T @ (0.5 * (Kr + Kr.T)) @ T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="8,16,24")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    freqs = np.concatenate([TRAIN, HELD])
    all_band = Band(p, cav, f, freqs)
    hw = lambda w: TAN * w  # resonance half-width in omega^2
    print(f"true lambda_1 = {lam1:.3f} (double), half-width there = {hw(lam1):.3f}, band {BAND}")

    rows = []
    for loss in ("galerkin", "proj", "mixed"):
        print(f"\nLoss: {loss}")
        for n in sizes:
            t0 = time.time()
            runs = [train(tr, n, loss, kind, seed, a.steps) for kind, seed in STARTS]
            b = min(runs, key=lambda r: r["final"])
            with torch.no_grad():
                Q = oracle.lifted_basis(p, b["logits"])
                ev = evaluate(tr, he, Q)
                gal, _, _ = band_errors(all_band, Q)
            eig = reduced_eigs(p, Q)
            phys = eig[eig > ZERO]
            in_band = phys[(phys >= BAND[0]) & (phys <= BAND[1])]
            near_true = float(np.min(abs(phys - lam1)))
            present = near_true < hw(lam1)
            spurious = in_band[abs(in_band - lam1) >= hw(lam1)]
            d_half = np.array([np.min(abs(phys - w)) / hw(w) for w in freqs])
            gal = gal.numpy()
            bad = gal > 0.5
            print(f"  n={n:2d}  reproduction: held gal max {ev['held_gal_max']:.3f}, proj max {ev['held_proj_max']:.3f}"
                  f" | true mode within half-width: {present} (nearest {phys[np.argmin(abs(phys - lam1))]:.3f})"
                  f" | in band: {np.round(in_band, 2).tolist()} | spurious in band: {len(spurious)}"
                  f" | freqs with gal>0.5: {int(bad.sum())}/{len(freqs)}"
                  f" (median dist to nearest eig {np.median(d_half[bad]) if bad.any() else float('nan'):.1f} half-widths"
                  f" vs {np.median(d_half[~bad]) if (~bad).any() else float('nan'):.1f} when gal<=0.5)"
                  f" ({time.time() - t0:.0f}s)", flush=True)
            rows.append({
                "loss": loss, "n": n, "held_gal_max": ev["held_gal_max"], "held_proj_max": ev["held_proj_max"],
                "true_mode_present": present, "nearest_to_true": float(phys[np.argmin(abs(phys - lam1))]),
                "n_in_band": len(in_band), "n_spurious_in_band": len(spurious),
                "in_band": str(np.round(in_band, 3).tolist()), "n_bad_freqs": int(bad.sum()),
                "median_halfwidths_bad": float(np.median(d_half[bad])) if bad.any() else np.nan,
                "median_halfwidths_ok": float(np.median(d_half[~bad])) if (~bad).any() else np.nan,
            })

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"diag_band_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
