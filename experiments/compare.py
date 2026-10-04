"""Train and score every method on the same instances. Usage: uv run python -m experiments.compare"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from baselines import REGISTRY
from compat_galerkin import metrics
from compat_galerkin.device import get_device
from compat_galerkin.problems import Family, FamilyConfig

LEVELS = (1, 2, 3)
KEYS = ["rel_err", "residual", "power_balance", "pec_violation", "energy_err", "gauss_fine"] + [f"gauss_r{l}" for l in LEVELS]


def run(methods, refine, grid, ntrain, ntest, seed, device, fno_epochs):
    fam = Family(FamilyConfig(refine=refine, grid=grid))
    t0 = time.time()
    train, test = fam.dataset(ntrain, seed), fam.dataset(ntest, seed + 1)
    print(f"mesh: {fam.base.n_edges} edges | data: {ntrain} train / {ntest} test in {time.time()-t0:.1f}s | device: {device}")
    hats = {l: fam.coarse_hats(l) for l in LEVELS}
    rows = []
    for name in methods:
        m = REGISTRY[name]()
        if name == "fno":
            m.epochs = fno_epochs
        t0 = time.time()
        info = m.fit(fam, train, device=device)
        fit_s = time.time() - t0
        per, times = [], []
        for inst, e_ref in zip(test.instances, test.e_ref):
            cav = fam.cavity(inst)
            t0 = time.perf_counter()
            e = m.predict(fam, inst, cav)
            times.append(time.perf_counter() - t0)
            per.append(metrics.evaluate(cav, inst, e, e_ref, hats))
        agg = {k: float(np.median([p[k] for p in per])) for k in KEYS}
        agg["rel_err_p90"] = float(np.percentile([p["rel_err"] for p in per], 90))
        rows.append({"method": name, "dim": info.get("dim", ""), "fit_s": round(fit_s, 2),
                     "predict_ms": round(1e3 * float(np.median(times)), 2), **agg})
        print(f"done {name}: {info}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="fine_fem,coarse_r1,coarse_r2,pod_16,pod_64,fno")
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--ntrain", type=int, default=200)
    ap.add_argument("--ntest", type=int, default=50)
    ap.add_argument("--fno-epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = run(a.methods.split(","), a.refine, a.grid, a.ntrain, a.ntest, a.seed, get_device(a.device), a.fno_epochs)
    cols = ["method", "dim", "fit_s", "predict_ms", "rel_err", "rel_err_p90", "residual", "power_balance",
            "pec_violation", "energy_err", "gauss_fine"] + [f"gauss_r{l}" for l in LEVELS]
    for block in (cols[:8], ["method"] + cols[8:]):
        print("\n" + "  ".join(f"{c:>14}" for c in block))
        for r in rows:
            print("  ".join(f"{r[c]:>14.3e}" if isinstance(r[c], float) else f"{r[c]!s:>14}" for c in block))
    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"compare_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
