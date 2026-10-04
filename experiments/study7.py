"""Study 7: is the encoder under-fitted, and do projection pre-training and more bumps help?

Four configurations with a total of 1500 optimiser steps (G8, P8, G12, P12), 100 and 400 training instances, three
repeats, the Gate 2 test sets. Protocol and decision rules: README "Study 7 protocol". Usage:
    OMP_NUM_THREADS=8 uv run python -m experiments.study7 [--repeats 3] [--sizes 100,400]
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np

from baselines import EncoderWhitney
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.gate1b_geometry import evaluate_method, score
from experiments.study4a import selected_config

CONFIGS = {
    "G8": (8, [("galerkin", 1500)]),
    "P8": (8, [("projection", 500), ("galerkin", 1000)]),
    "G12": (12, [("galerkin", 1500)]),
    "P12": (12, [("projection", 500), ("galerkin", 1000)]),
}
THRESH = 0.10


def committed(pattern, method, key, n_train):
    """Mean over repeats of `key` for a method and training size in the latest CSV that matches `pattern`."""
    rows = list(csv.DictReader(open(sorted(glob.glob(pattern))[-1])))
    return float(np.mean([float(r[key]) for r in rows if r["method"] == method and int(r["n_train"]) == n_train]))


def write(rows, out, stem):
    Path(out).mkdir(exist_ok=True)
    path = Path(out) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = list(dict.fromkeys(k for x in rows for k in x))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", default="100,400")
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--step-scale", type=float, default=1.0, help="scales every stage length (smoke tests only)")
    ap.add_argument("--configs", default="G8,P8,G12,P12")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    names = a.configs.split(",")
    width, _ = selected_config()
    fam = Family(FamilyConfig(refine=4, grid=64))
    print(f"width {width}; configurations {names}; step scale {a.step_scale}", flush=True)

    rows = []
    for rep in range(a.repeats):
        t0 = time.time()
        train_all, test = fam.dataset(max(sizes), seed=400 + rep), fam.dataset(a.ntest, seed=500 + rep)
        ratios = np.array([i.params["ratio"] for i in test.instances])
        for n in sizes:
            train = Dataset(train_all.instances[:n], train_all.e_ref[:n])
            for name in names:
                bumps, sched = CONFIGS[name]
                sched = [(k, max(2, int(s * a.step_scale))) for k, s in sched]
                m = EncoderWhitney(n=bumps, width=width, seed=rep, schedule=sched)
                info = m.fit(fam, train)
                s = score(evaluate_method(fam, m, test), ratios)
                proj = m.projection_errors(fam, test)
                rows.append({"repeat": rep, "config": name, "n_train": n, "bumps": bumps, "dim": info["dim"], "params": info["params"],
                             "train_error": info["train_error"], "train_median": info["train_median"],
                             "median_field": s["median_field"], "median_energy": s["median_energy"], "frac_energy5": s["frac_energy5"],
                             "median_projection": float(np.median(proj)), "fit_seconds": info["fit_seconds"]})
                print(f"  [{rep}] n_train={n:>3} {name:<4} dim {info['dim']:>3}: train {info['train_error']:.3f} | test median field "
                      f"{s['median_field']:.3f} energy {s['median_energy']:.3f} | best-approx (projection) {np.median(proj):.3f} "
                      f"({info['fit_seconds']:.0f}s)", flush=True)
        print(f"repeat {rep}: {time.time() - t0:.0f}s", flush=True)
        write(rows, a.out, "study7_partial")

    mean = lambda name, n, key="median_field": float(np.mean([r[key] for r in rows if r["config"] == name and r["n_train"] == n]))
    g2 = {n: committed("results/gate2_2*.csv", "encoder", "median_field", n) for n in sizes}
    g2_train = {n: committed("results/gate2_2*.csv", "encoder", "train_error", n) for n in sizes}
    fno = {n: committed("results/data_efficiency_2*.csv", "fno", "median_field", n) for n in sizes}
    print("\nMean over repeats: median field error / training error / best-approximation error of the predicted space")
    print(f"  Gate 2 (1000 steps, n=8): " + ", ".join(f"{n}: {g2[n]:.3f}/{g2_train[n]:.3f}" for n in sizes) +
          " | FNO: " + ", ".join(f"{n}: {fno[n]:.3f}" for n in sizes))
    for name in names:
        print(f"  {name:<4}" + "  ".join(f"{n}: {mean(name, n):.3f}/{mean(name, n, 'train_error'):.3f}/{mean(name, n, 'median_projection'):.3f}" for n in sizes))

    def verdict(new, old, label):
        r = [mean(new, n) / mean(old, n) for n in sizes]
        if all(x <= 1 - THRESH for x in r):
            v = "HELPS"
        elif any(x >= 1 + THRESH for x in r):
            v = "HURTS"
        else:
            v = "NO EFFECT"
        print(f"  {label}: {v} (ratios {', '.join(f'{n}: {x:.2f}' for n, x in zip(sizes, r))})")

    print("\nRegistered comparisons (mean median field error ratio, new over old; help = at most 0.90 at both sizes, hurt = at least 1.10 at either)")
    if "G8" in names and 400 in sizes:
        x = mean("G8", 400) / g2[400]
        t = mean("G8", 400, "train_error") / g2_train[400]
        print(f"  More steps (G8 against Gate 2 at 400 instances): ratio {x:.2f} -> {'HELP' if x <= 1 - THRESH else 'NO CLEAR HELP'}")
        print(f"  Under-fitting supported (training error of G8 over Gate 2 at 400 instances: {t:.2f}) -> {'YES' if t <= 1 - THRESH else 'NO'}")
    if {"G8", "P8"} <= set(names):
        verdict("P8", "G8", "Pre-training, n = 8")
    if {"G12", "P12"} <= set(names):
        verdict("P12", "G12", "Pre-training, n = 12")
    if {"G8", "G12"} <= set(names):
        verdict("G12", "G8", "More bumps (dimension 36 -> 78)")
    closes = [name for name in names if all(mean(name, n) <= fno[n] for n in sizes)]
    print(f"  Closes the gap to the FNO at both sizes: {closes or 'none'}")
    write(rows, a.out, "study7")


if __name__ == "__main__":
    main()
