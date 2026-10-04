"""Band pre-check: how close to the first resonance can the oracle stay accurate?

Protocol and the band-selection rule are in the README ("Band pre-check protocol"). Usage:
    uv run python -m experiments.band_precheck [--steps 500] [--n 8]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from compat_galerkin import fem, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.gate1 import QOI_TOL, build_instance, frequencies

RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SELECTABLE_MAX = 0.8
MIN_RUN = 3


def select_band(passing):
    """Longest contiguous run of passing ratios among the selectable ones (ties: the lower run)."""
    sel = [r for r in RATIOS if r <= SELECTABLE_MAX]
    best, run = [], []
    for r in sel:
        if passing.get(r):
            run.append(r)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    return best if len(best) >= MIN_RUN else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    print(f"lambda_1 = {lam1:.3f}, n = {a.n}, steps = {a.steps}", flush=True)

    rows, passing = [], {}
    for r in RATIOS:
        t0 = time.time()
        inst = Instance(eps, r * lam1 * (1 + 0.05j), f, {})
        p = oracle.OracleProblem.from_instance(cav, inst, fem.solve_fine(cav, inst.omega2, f))
        runs = [oracle.optimise(p, a.n, kind, seed, a.steps) for kind, seed in [("rbf", 0), ("smooth", 0)]]
        b = min(runs, key=lambda x: x["final"])
        with torch.no_grad():
            row = {"ratio": r, "omega2": r * lam1, "rel_err": b["final"],
                   "proj_err": float(oracle.projection_error(p, b["logits"])),
                   "energy_err": float(oracle.energy_error(p, b["logits"])), "converged": b["converged"]}
        passing[r] = row["energy_err"] < QOI_TOL
        rows.append(row)
        tag = "pass" if passing[r] else "fail"
        note = "  (indication only, not selectable)" if r > SELECTABLE_MAX else ""
        print(f"  r={r:.1f} omega^2={r * lam1:5.2f}: field {row['rel_err']:.3f} proj {row['proj_err']:.3f} "
              f"energy {row['energy_err']:.3f} conv={row['converged']} -> {tag}{note} ({time.time() - t0:.0f}s)", flush=True)

    band = select_band(passing)
    if band is None:
        print("\nBand: NO RUN OF THREE PASSING RATIOS. Return the choice to the project owner.")
    else:
        print(f"\nBand: r in [{band[0]}, {band[-1]}] (ratios {band})")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"band_precheck_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
