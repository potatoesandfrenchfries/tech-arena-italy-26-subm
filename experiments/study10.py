"""Study 10: a grid-base, mesh-free encoder with multi-resolution training.

Variants GB (trained at refine 4) and GBMR (trained at refine 4 and 5); tested at refine 4 to 7 without retraining.
Protocol and decision rules: README "Study 10 protocol". Usage:
    OMP_NUM_THREADS=8 uv run python -u -m experiments.study10 [--repeats 3] [--sizes 100,400]
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np

from baselines import GridBaseEncoder
from compat_galerkin import metrics
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.study4a import selected_config
from experiments.study5c import EncoderOnMesh
from experiments.study6a import cached_dataset

PARITY, HOLD = 1.15, 1.25
FNO_R6_R7 = {100: 0.25, 400: 0.17}


def write(rows, out, stem):
    Path(out).mkdir(exist_ok=True)
    path = Path(out) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = list(dict.fromkeys(k for x in rows for k in x))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def g12_reference(sizes):
    """Mean median field error of the per-node 78-dimensional encoder at refine 4 (Study 8, stage B)."""
    rows = list(csv.DictReader(open(sorted(glob.glob("results/study8_transfer_2*.csv"))[-1])))
    return {n: float(np.mean([float(r["median_field"]) for r in rows if r["model"] == f"g12_n{n}" and int(r["refine"]) == 4])) for n in sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", default="100,400")
    ap.add_argument("--meshes", default="4,5,6,7")
    ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--variants", default="GB,GBMR")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes, meshes, variants = [int(x) for x in a.sizes.split(",")], [int(x) for x in a.meshes.split(",")], a.variants.split(",")
    width, _ = selected_config()
    fams = {r: Family(FamilyConfig(refine=r, grid=64)) for r in set(meshes) | {4, 5}}
    tests = {r: cached_dataset(fams[r], r, a.ntest, 500) for r in meshes}
    print(f"width {width}, steps {a.steps}, variants {variants}", flush=True)

    rows = []
    for rep in range(a.repeats):
        t0 = time.time()
        train4 = fams[4].dataset(max(sizes), seed=400 + rep)
        train5 = fams[5].dataset(max(sizes), seed=400 + rep) if "GBMR" in variants else None
        for n in sizes:
            d4 = Dataset(train4.instances[:n], train4.e_ref[:n])
            for v in variants:
                enc = GridBaseEncoder(n=12, width=width, seed=rep, schedule=[("galerkin", a.steps)])
                if v == "GB":
                    info = enc.fit(fams[4], d4)
                else:
                    info = enc.fit_multi([(fams[4], d4), (fams[5], Dataset(train5.instances[:n], train5.e_ref[:n]))])
                print(f"  [{rep}] {v} n_train={n}: train error {info['train_error']:.3f}, {info['params']} parameters ({info['fit_seconds']:.0f}s)", flush=True)
                model = EncoderOnMesh(enc, enc.p0.coords, fams[4].base.mesh, "nearest")
                for r in meshes:
                    fam, ds = fams[r], tests[r]
                    errs = []
                    for inst, e_ref in zip(ds.instances, ds.e_ref):
                        ev = metrics.evaluate(fam.cavity(inst), inst, model.predict(fam, inst, fam.cavity(inst)), e_ref)
                        errs.append((ev["rel_err"], ev["energy_err"]))
                    arr = np.array(errs)
                    rows.append({"repeat": rep, "variant": v, "n_train": n, "refine": r, "median_field": float(np.median(arr[:, 0])),
                                 "median_energy": float(np.median(arr[:, 1])), "train_error": info["train_error"], "fit_seconds": info["fit_seconds"]})
                print(f"  [{rep}] {v} n_train={n}: " + ", ".join(f"refine {r} {x['median_field']:.3f}" for r, x in
                                                                 ((x['refine'], x) for x in rows[-len(meshes):])) + f" ({time.time() - t0:.0f}s)", flush=True)
        write(rows, a.out, "study10_partial")

    mean = lambda v, n, r, key="median_field": float(np.mean([x[key] for x in rows if x["variant"] == v and x["n_train"] == n and x["refine"] == r]))
    ref = g12_reference(sizes)
    print("\nMedian field error by mesh (mean over repeats); per-node G12 at refine 4: " + ", ".join(f"{n}: {ref[n]:.3f}" for n in sizes))
    outcomes = {}
    for v in variants:
        parity, holds = [], []
        for n in sizes:
            vals = {r: mean(v, n, r) for r in meshes}
            ratios = {r: vals[r] / vals[4] for r in meshes if r != 4}
            spread = [x["median_field"] for x in rows if x["variant"] == v and x["n_train"] == n and x["refine"] == 4]
            parity.append(vals[4] <= PARITY * ref[n])
            holds.append(all(ratios[r] <= HOLD for r in (6, 7) if r in ratios))
            print(f"  {v:<5} n_train={n}: " + "  ".join(f"refine {r}: {vals[r]:.3f}" + (f" ({ratios[r]:.2f})" if r != 4 else "") for r in meshes)
                  + f" | refine-4 range over repeats {max(spread) - min(spread):.3f}{' UNSTABLE' if max(spread) - min(spread) > 0.1 else ''}"
                  + f" | parity {'yes' if parity[-1] else 'no'}, transfer {'holds' if holds[-1] else 'fails'}")
        outcomes[v] = ("SOLVES" if all(parity) and all(holds) else "TRANSFERS BUT LOSES ACCURACY" if all(holds) else
                       "ACCURATE BUT DOES NOT TRANSFER" if all(parity) else "FAILS")
    print("\nOutcomes: " + ", ".join(f"{v}: {o}" for v, o in outcomes.items()))
    print("Readout: FNO median field error at refine 6 and 7 is about " + ", ".join(f"{n}: {FNO_R6_R7[n]}" for n in sizes if n in FNO_R6_R7))
    write(rows, a.out, "study10")


if __name__ == "__main__":
    main()
