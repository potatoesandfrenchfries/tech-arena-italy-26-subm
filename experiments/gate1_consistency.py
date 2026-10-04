"""Gate 1 follow-up: how consistent are the oracle-W solutions at fine-mesh level?

The README inference is that the weak Gauss law holds only against the gradients of the learned
functions W_i, not against all fine-mesh nodes. This measures it on the oracle W of the Gate 1
passing cases, with the same Gauss metrics used for the baselines:

  gauss_W      test functions = the n learned bumps W_i (the guaranteed set)
  gauss_r1..3  test functions = hat functions of the level-L coarse meshes
  gauss_fine   test functions = all interior fine nodes (the true divergence error)

Reference rows: the fine solution, and the non-learned coarse Whitney spaces.

    uv run python -m experiments.gate1_consistency [--steps 500]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from baselines import CoarseWhitney
from compat_galerkin import fem, lift, metrics, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.gate1 import STARTS, build_instance, frequencies

CASES = {"low": [4, 8, 16], "gap": [8, 16]}  # Gate 1 passing frequencies and sizes
COLS = ["gauss_W", "gauss_r1", "gauss_r2", "gauss_r3", "gauss_fine", "power_balance", "rel_err"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    freqs, _ = frequencies(cav)
    hats = {lvl: fam.coarse_hats(lvl) for lvl in (1, 2, 3)}
    idx = cav.interior_edges

    rows = []

    def record(freq, label, n, dim, e, e_ref, inst, W=None):
        h = dict(hats)
        if W is not None:
            h["W"] = W
        ev = metrics.evaluate(cav, inst, e, e_ref, h)
        ev["gauss_W"] = ev.get("gauss_rW", np.nan)  # evaluate names test sets gauss_r{key}
        row = {"freq": freq, "space": label, "n": n, "dim": dim, **{c: ev.get(c, np.nan) for c in COLS}}
        rows.append(row)
        print(f"  {label:<14} dim={dim:4d} " + " ".join(f"{c}={row[c]:.1e}" for c in COLS), flush=True)

    for fname, sizes in CASES.items():
        inst = Instance(eps, freqs[fname] * (1 + 0.05j), f, {})
        e_ref = fem.solve_fine(cav, inst.omega2, f)
        p = oracle.OracleProblem.from_instance(cav, inst, e_ref)
        print(f"\n[{fname}] omega^2 = {freqs[fname]:.3f}")
        record(fname, "fine solution", 0, len(idx), e_ref, e_ref, inst)
        for r in (1, 2, 3):
            m = CoarseWhitney(r)
            info = m.fit(fam, None)
            record(fname, f"coarse_r{r}", 0, info["dim"], m.predict(fam, inst, cav), e_ref, inst)
        for n in sizes:
            t0 = time.time()
            runs = [oracle.optimise(p, n, kind, seed, a.steps) for kind, seed in STARTS]
            lg = runs[int(np.argmin([r["final"] for r in runs]))]["logits"]
            with torch.no_grad():
                Q = oracle.lifted_basis(p, lg)
                W = lift.make_pou(lg, p.boundary_mask)[:, :n].numpy()
                e = np.zeros(cav.n_edges, dtype=complex)
                e[idx] = lift.reduced_solve(Q, p.K, p.M, p.f, p.omega2).numpy()
            record(fname, f"oracle n={n}", n, Q.shape[1], e, e_ref, inst, W)
            print(f"    ({time.time() - t0:.0f}s)")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1_consistency_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
