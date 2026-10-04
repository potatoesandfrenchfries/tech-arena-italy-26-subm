"""Baseline data-efficiency study: FNO and POD-Galerkin error against training-set size.

Protocol is in the README ("Baseline data-efficiency study protocol"). Usage (FNO needs the GPU):
    uv run python -m experiments.data_efficiency [--repeats 3] [--sizes 25,50,100,200,400]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from baselines import FNO, PODGalerkin
from compat_galerkin.device import get_device
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.gate1b_geometry import ENERGY_TOL, FIELD_TOL, evaluate_method, score

POD_DIMS = (36, 136)


def first_below(rows, method, key, tol):
    """Smallest training size whose mean (over repeats) of `key` is below tol, or None."""
    sizes = sorted({r["n_train"] for r in rows if r["method"] == method})
    for n in sizes:
        v = np.mean([r[key] for r in rows if r["method"] == method and r["n_train"] == n])
        if v < tol:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", default="25,50,100,200,400")
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    device = get_device("auto")
    print(f"device {device}, grid {a.grid}, epochs {a.epochs}", flush=True)

    fam = Family(FamilyConfig(refine=4, grid=a.grid))
    rows = []
    for r in range(a.repeats):
        t0 = time.time()
        train_all = fam.dataset(max(sizes), seed=400 + r)
        test = fam.dataset(a.ntest, seed=500 + r)
        ratios = np.array([i.params["ratio"] for i in test.instances])
        for n in sizes:
            train = Dataset(train_all.instances[:n], train_all.e_ref[:n])
            models = {"fno": FNO(epochs=a.epochs, batch=16, seed=r)}
            for dim in POD_DIMS:
                models[f"pod{dim}"] = PODGalerkin(rank=dim)
            for name, m in models.items():
                info = m.fit(fam, train, device=device)
                s = score(evaluate_method(fam, m, test), ratios)
                row = {"repeat": r, "method": name, "n_train": n, "dim": info.get("dim", info.get("params")), **s}
                if name == "fno":
                    row.update(roundtrip_floor=info["roundtrip_floor"], fit_seconds=info["fit_seconds"],
                               final_train_loss=info["final_train_loss"])
                rows.append(row)
                print(f"  [{r}] n_train={n:>3} {name:<7} median field {s['median_field']:.3f} mean {s['mean_field']:.3f} | "
                      f"median energy {s['median_energy']:.3f} | energy<5%: {s['frac_energy5']:.2f} field<10%: {s['frac_field10']:.2f}"
                      + (f" | floor {info['roundtrip_floor']:.3f}, {info['fit_seconds']:.0f}s" if name == "fno" else ""), flush=True)
        print(f"repeat {r}: {time.time() - t0:.0f}s", flush=True)

    print("\nMean over repeats: median field error / median energy error")
    for name in ["fno"] + [f"pod{d}" for d in POD_DIMS]:
        line = ", ".join(
            f"{n}: {np.mean([x['median_field'] for x in rows if x['method'] == name and x['n_train'] == n]):.3f}/"
            f"{np.mean([x['median_energy'] for x in rows if x['method'] == name and x['n_train'] == n]):.3f}" for n in sizes)
        print(f"  {name:<7} {line}")
    print("\nSmallest training size with mean median-field error below 10% / mean median-energy error below 5%:")
    for name in ["fno"] + [f"pod{d}" for d in POD_DIMS]:
        print(f"  {name:<7} field: {first_below(rows, name, 'median_field', FIELD_TOL)}, energy: "
              f"{first_below(rows, name, 'median_energy', ENERGY_TOL)}")
    print("\nFNO / POD-136 median field error ratio by size: " + ", ".join(
        f"{n}: {np.mean([x['median_field'] for x in rows if x['method'] == 'fno' and x['n_train'] == n]) / np.mean([x['median_field'] for x in rows if x['method'] == 'pod136' and x['n_train'] == n]):.2f}"
        for n in sizes))

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"data_efficiency_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = list(dict.fromkeys(k for x in rows for k in x))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
