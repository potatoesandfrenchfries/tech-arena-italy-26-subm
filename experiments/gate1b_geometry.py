"""Gate 1b stage (b): does one shared W work across geometries and frequencies in the sub-resonant band?

Compares a global shared W (no encoder), POD of the same dimension, the non-learned coarse Whitney spaces,
and a per-instance oracle W (upper bound that shows the headroom an encoder could use).
Protocol and decision rules are in the README ("Gate 1b stage (b) protocol"). Usage:
    uv run python -m experiments.gate1b_geometry [--repeats 3] [--ntrain 200] [--ntest 300]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from baselines import CoarseWhitney, PODGalerkin, SharedWhitney
from compat_galerkin import fem, metrics, oracle
from compat_galerkin.problems import Family, FamilyConfig

BUDGET, ENERGY_TOL, FIELD_TOL, FRACTION = 150, 0.05, 0.10, 0.8
BINS = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
ORACLE_STARTS = (("rbf", 0), ("smooth", 0))


def score(errs, ratios):
    """Summary of per-instance (field error, energy error) arrays."""
    rel, en = np.array([e[0] for e in errs]), np.array([e[1] for e in errs])
    out = {"median_field": float(np.median(rel)), "mean_field": float(rel.mean()), "median_energy": float(np.median(en)),
           "frac_energy5": float(np.mean(en < ENERGY_TOL)), "frac_field10": float(np.mean(rel < FIELD_TOL))}
    for lo, hi in BINS:
        m = (ratios >= lo) & (ratios < hi + (1e-12 if hi >= 0.6 else 0))
        out[f"median_field_bin_{lo}_{hi}"] = float(np.median(rel[m])) if m.any() else float("nan")
    return out


def evaluate_method(fam, method, ds):
    errs = []
    for inst, e_ref in zip(ds.instances, ds.e_ref):
        cav = fam.cavity(inst)
        ev = metrics.evaluate(cav, inst, method.predict(fam, inst, cav), e_ref)
        errs.append((ev["rel_err"], ev["energy_err"]))
    return errs


def oracle_errors(fam, ds, n, steps, count):
    """Per-instance oracle W: best of two starts per instance, evaluated on the first `count` instances."""
    errs = []
    for inst, e_ref in list(zip(ds.instances, ds.e_ref))[:count]:
        p = oracle.OracleProblem.from_instance(fam.cavity(inst), inst, e_ref)
        runs = [oracle.optimise(p, n, kind, seed, steps) for kind, seed in ORACLE_STARTS]
        b = min(runs, key=lambda r: r["final"])
        with torch.no_grad():
            errs.append((float(oracle.galerkin_error(p, b["logits"])), float(oracle.energy_error(p, b["logits"]))))
    return errs


def passes(row):
    return row["dim"] <= BUDGET and row["median_energy"] < ENERGY_TOL and row["frac_energy5"] >= FRACTION


def verdict(rows):
    def mean_rows(method, n):
        rs = [r for r in rows if r["method"] == method and r["n"] == n]
        return None if not rs else {k: float(np.mean([r[k] for r in rs])) if isinstance(rs[0][k], (int, float)) else rs[0][k]
                                    for k in rs[0]}

    sizes = sorted({r["n"] for r in rows if r["method"] == "shared"})
    shared = {n: mean_rows("shared", n) for n in sizes}
    pod = {n: mean_rows("pod", n) for n in sizes}
    ora = {n: mean_rows("oracle", n) for n in sorted({r["n"] for r in rows if r["method"] == "oracle"})}
    lines = []
    ratios = {n: shared[n]["median_field"] / pod[n]["median_field"] for n in sizes}
    better = sum(r <= 0.5 for r in ratios.values())
    worse = sum(r > 2 for r in ratios.values())
    lines.append("shared W / POD median field error at equal dimension: " + ", ".join(f"n={n}: {r:.2f}" for n, r in ratios.items()))
    lines.append("structure vs POD: " + ("HELPS (<= 0.5x at two or more sizes)" if better >= 2 else
                                         "WORSE (> 2x at two or more sizes)" if worse >= 2 else "COMPARABLE"))
    if any(passes(shared[n]) for n in sizes):
        out = "GLOBAL W SUFFICES"
    elif any(passes(ora[n]) for n in ora):
        out = "ENCODER NEEDED (shared W fails, per-instance oracle passes)"
    else:
        out = "NO HEADROOM (neither the shared W nor the per-instance oracle passes)"
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ntrain", type=int, default=200)
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sizes", default="4,8,12,16")
    ap.add_argument("--oracle-count", type=int, default=20)
    ap.add_argument("--oracle-sizes", default="8,16")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    oracle_sizes = [int(x) for x in a.oracle_sizes.split(",")]

    fam = Family(FamilyConfig(refine=4))
    coarse = {}
    for lvl in (1, 2, 3):
        m = CoarseWhitney(lvl)
        info = m.fit(fam, None)
        coarse[lvl] = (m, info["dim"])

    rows = []
    for r in range(a.repeats):
        t0 = time.time()
        train, test = fam.dataset(a.ntrain, seed=400 + r), fam.dataset(a.ntest, seed=500 + r)
        ratios = np.array([i.params["ratio"] for i in test.instances])
        per = {}

        def record(method, n, dim, errs, extra=None):
            per[(method, n)] = errs
            rows.append({"repeat": r, "method": method, "n": n, "dim": dim, **score(errs, ratios[: len(errs)]), **(extra or {})})
            s = rows[-1]
            print(f"  [{r}] {method:<8} n={n:>2} dim={dim:>4}: median field {s['median_field']:.3f} mean {s['mean_field']:.3f} "
                  f"| median energy {s['median_energy']:.3f} | energy<5%: {s['frac_energy5']:.2f} field<10%: {s['frac_field10']:.2f}", flush=True)

        for lvl, (m, dim) in coarse.items():
            record(f"coarse_r{lvl}", lvl, dim, evaluate_method(fam, m, test))
        for n in sizes:
            sw = SharedWhitney(n=n, steps=a.steps, batch=a.batch)
            info = sw.fit(fam, train)
            record("shared", n, info["dim"], evaluate_method(fam, sw, test), {"train_error": info["train_error"]})
            pod = PODGalerkin(rank=n * (n + 1) // 2)
            pinfo = pod.fit(fam, train)
            record("pod", n, pinfo["dim"], evaluate_method(fam, pod, test))
        if r == 0:
            for n in oracle_sizes:
                errs = oracle_errors(fam, test, n, a.steps, a.oracle_count)
                record("oracle", n, n * (n + 1) // 2, errs)
                for method in ("shared", "pod"):  # the same instances, for the headroom comparison
                    if (method, n) in per:
                        sub = per[(method, n)][: len(errs)]
                        rows.append({"repeat": r, "method": f"{method}_on_oracle_subset", "n": n, "dim": n * (n + 1) // 2,
                                     **score(sub, ratios[: len(sub)])})
                        s = rows[-1]
                        print(f"  [{r}] {method + ' (oracle subset)':<22} n={n:>2}: median field {s['median_field']:.3f} "
                              f"| median energy {s['median_energy']:.3f} | energy<5%: {s['frac_energy5']:.2f}", flush=True)
        print(f"repeat {r}: {time.time() - t0:.0f}s", flush=True)

    out, lines = verdict(rows)
    print("\nSummary (mean over repeats), median field error by method and size:")
    for method in ["coarse_r1", "coarse_r2", "coarse_r3", "shared", "pod", "oracle"]:
        for n in sorted({x["n"] for x in rows if x["method"] == method}):
            rs = [x for x in rows if x["method"] == method and x["n"] == n]
            print(f"  {method:<10} n={n:>2} dim={rs[0]['dim']:>4}: median field {np.mean([x['median_field'] for x in rs]):.3f}, "
                  f"median energy {np.mean([x['median_energy'] for x in rs]):.3f}, energy<5% {np.mean([x['frac_energy5'] for x in rs]):.2f}")
    for line in lines:
        print(line)
    print(f"\nVerdict: {out}")

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"gate1b_geometry_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = list(dict.fromkeys(k for x in rows for k in x))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
