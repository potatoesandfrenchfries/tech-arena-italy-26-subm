"""Gate 1 (capacity oracle). Protocol and decision rules are in the README. Usage:
    uv run python -m experiments.gate1 [--steps 500] [--sizes 4,6,8,12,16,24]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

from baselines import CoarseWhitney
from compat_galerkin import fem, lift, metrics, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance

BUDGET_FRAC, QOI_TOL, PLATEAU = 0.10, 0.05, 0.8
STARTS = [("rbf", 0), ("rbf", 1), ("smooth", 0), ("smooth", 1)]


def build_instance(fam):
    mesh = fam.base.mesh
    eps = fem.inclusion_eps(mesh, 4.0, (0.5, 0.5), 0.2)
    cav = fam.base.with_eps(eps)
    f = fem.current_source(cav, (0.3, 0.4), 0.08, (0.0, 1.0))
    return cav, eps, f


def frequencies(cav):
    w = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray(), eigvals_only=True)
    phys = w[w > 1e-6]
    modes = [phys[0]]
    for x in phys[1:]:
        if abs(x - modes[-1]) / modes[-1] > 1e-3:
            modes.append(x)
        if len(modes) == 2:
            break
    return {"low": 0.4 * modes[0], "near": 1.02 * modes[0], "gap": 0.5 * (modes[0] + modes[1])}, modes


def numerical_rank(p, logits):
    W = lift.make_pou(logits, p.boundary_mask)
    V = lift.whitney_lift(W, p.A, p.G)[p.idx]
    s = torch.linalg.svdvals(V)
    return int((s > 1e-8 * s[0]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sizes", default="4,6,8,12,16,24")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    fam = Family(FamilyConfig(refine=4))
    cav0, eps, f = build_instance(fam)
    freqs, modes = frequencies(cav0)
    budget = int(BUDGET_FRAC * len(cav0.interior_edges))
    print(f"interior edges {len(cav0.interior_edges)} | budget dim <= {budget} | modes {np.round(modes, 3)}")
    print({k: round(v, 3) for k, v in freqs.items()})

    rows, verdicts = [], {}
    for fname, w2 in freqs.items():
        inst = Instance(eps, w2 * (1 + 0.05j), f, {})
        e_ref = fem.solve_fine(cav0, inst.omega2, f)
        p = oracle.OracleProblem.from_instance(cav0, inst, e_ref)
        ctx = []
        for r in (1, 2, 3):
            m = CoarseWhitney(r)
            info = m.fit(fam, None)
            ev = metrics.evaluate(cav0, inst, m.predict(fam, inst, cav0), e_ref)
            ctx.append({"freq": fname, "method": f"coarse_r{r}", "dim": info["dim"],
                        "rel_err": ev["rel_err"], "energy_err": ev["energy_err"]})
        best_by_n = {}
        for n in sizes:
            t0 = time.time()
            runs = [oracle.optimise(p, n, kind, seed, a.steps) for kind, seed in STARTS]
            finals = np.array([r["final"] for r in runs])
            k = int(np.argmin(finals))
            lg = runs[k]["logits"]
            with torch.no_grad():
                row = {"freq": fname, "method": "oracle", "n": n, "dim": numerical_rank(p, lg),
                       "rel_err": float(finals[k]), "start_median": float(np.median(finals)),
                       "start_worst": float(finals.max()), "proj_err": float(oracle.projection_error(p, lg)),
                       "energy_err": float(oracle.energy_error(p, lg)), "converged": runs[k]["converged"]}
            rows.append(row)
            best_by_n[n] = row
            print(f"[{fname}] n={n:3d} dim={row['dim']:4d} rel_err={row['rel_err']:.3f} proj={row['proj_err']:.3f} "
                  f"energy={row['energy_err']:.3f} spread=({finals.min():.3f},{finals.max():.3f}) "
                  f"conv={row['converged']} ({time.time()-t0:.0f}s)", flush=True)
        rows += ctx

        ok = [r for r in best_by_n.values() if r["dim"] <= budget and r["energy_err"] < QOI_TOL]
        if ok:
            v = f"PASS at n={min(r['n'] for r in ok)}"
        elif not all(r["converged"] for r in best_by_n.values()):
            v = "INCONCLUSIVE (optimiser not converged)"
        elif 16 in best_by_n and 24 in best_by_n and best_by_n[24]["rel_err"] >= PLATEAU * best_by_n[16]["rel_err"]:
            v = "FAIL (capacity: plateaued)"
        else:
            v = "INCONCLUSIVE (still improving at the largest n)"
        verdicts[fname] = v

    print("\nContext (non-learned compatible spaces, same instance):")
    for r in rows:
        if r["method"].startswith("coarse"):
            print(f"  [{r['freq']}] {r['method']} dim={r['dim']} rel_err={r['rel_err']:.3f} energy_err={r['energy_err']:.3f}")
    print("\nVerdicts:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = ["freq", "method", "n", "dim", "rel_err", "start_median", "start_worst", "proj_err", "energy_err", "converged"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
