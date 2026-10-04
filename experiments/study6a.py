"""Study 6a (variant A2): does a mesh-free encoder transfer to finer meshes?

The encoder has no per-node parameters (analytic RBF base with trained centres and widths, plus the CNN correction), so
it is evaluated on any mesh by passing that mesh's node coordinates. Protocol and decision rules: README "Study 6a
protocol". Usage:
    OMP_NUM_THREADS=8 uv run python -m experiments.study6a [--repeats 3] [--meshes 4,5,6,7] [--ntest 200]
"""
import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from baselines import MeshFreeEncoder
from compat_galerkin import metrics
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.study4a import selected_config
from experiments.study5c import EncoderOnMesh

HOLD, PARITY = 1.25, 1.15
PER_NODE = {100: 0.242, 400: 0.180}  # refine-4 median field error of the per-node encoder (Study 5a)
NEAREST_R6 = {100: 0.822, 400: 0.781}  # Study 5a encoder at refine 6


def cached_dataset(fam, refine, n, seed, cache="results/cache"):
    """Dataset with full-wave references, cached on disk (the refine-7 references take about six minutes)."""
    path = Path(cache) / f"ds_r{refine}_n{n}_s{seed}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    ds = fam.dataset(n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ds


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
    ap.add_argument("--meshes", default="4,5,6,7")
    ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes, meshes = [int(x) for x in a.sizes.split(",")], [int(x) for x in a.meshes.split(",")]
    width, steps = selected_config()
    print(f"mesh-free encoder width {width}, steps {steps}", flush=True)

    fams, tests = {}, {}
    for r in meshes:
        t0 = time.time()
        fams[r] = Family(FamilyConfig(refine=r, grid=64))
        tests[r] = cached_dataset(fams[r], r, a.ntest, 500)
        print(f"refine {r}: {len(fams[r].base.interior_edges)} unknowns, {a.ntest} test instances ({time.time() - t0:.0f}s)", flush=True)

    Path(a.out, "models").mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(a.repeats):
        t0 = time.time()
        fam4 = fams[4]
        train_all = fam4.dataset(max(sizes), seed=400 + rep)
        models = {}
        for n in sizes:
            enc = MeshFreeEncoder(n=8, steps=steps, width=width, seed=rep)
            info = enc.fit(fam4, Dataset(train_all.instances[:n], train_all.e_ref[:n]))
            torch.save({"net": enc.net.state_dict(), "centres": enc.centres, "log_s": enc.log_s, "mu": enc.mu, "sd": enc.sd,
                        "n": 8, "width": width, "steps": steps, "n_train": n, "repeat": rep, "train_seed": 400 + rep},
                       Path(a.out) / "models" / f"mesh_free_encoder_n{n}_rep{rep}.pt")
            models[n] = EncoderOnMesh(enc, enc.p0.coords, fam4.base.mesh, "nearest")
            print(f"  [{rep}] n_train={n}: train error {info['train_error']:.3f}, {info['params']} parameters", flush=True)
        print(f"[{rep}] trained {len(sizes)} mesh-free encoders ({time.time() - t0:.0f}s)", flush=True)
        for r in meshes:
            fam, ds = fams[r], tests[r]
            errs = {n: [] for n in models}
            for inst, e_ref in zip(ds.instances, ds.e_ref):
                cav = fam.cavity(inst)
                for n, m in models.items():
                    ev = metrics.evaluate(cav, inst, m.predict(fam, inst, cav), e_ref)
                    errs[n].append((ev["rel_err"], ev["energy_err"]))
            for n, e in errs.items():
                arr = np.array(e)
                rows.append({"repeat": rep, "n_train": n, "refine": r, "median_field": float(np.median(arr[:, 0])),
                             "median_energy": float(np.median(arr[:, 1])), "frac_field10": float(np.mean(arr[:, 0] < 0.10))})
            print(f"  [{rep}] refine {r}: " + ", ".join(f"n={n} {np.median(np.array(e)[:, 0]):.3f}" for n, e in errs.items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
        write(rows, a.out, "study6a_partial")

    mean = lambda n, r, key="median_field": float(np.mean([x[key] for x in rows if x["n_train"] == n and x["refine"] == r]))
    print("\nMedian field error / median energy error (mean over repeats), mesh-free encoder")
    for n in sizes:
        print(f"  n_train={n}  " + "  ".join(f"refine {r}: {mean(n, r):.3f}/{mean(n, r, 'median_energy'):.3f}" for r in meshes))
    print(f"\nRatio to the refine-4 error (hold if at most {HOLD})")
    holds, ok = {}, True
    for n in sizes:
        ratios = {r: mean(n, r) / mean(n, 4) for r in meshes if r != 4}
        holds[n] = ratios
        print(f"  n_train={n}  " + "  ".join(f"refine {r}: {x:.2f}" for r, x in ratios.items()))
    repairs = all(holds[n].get(r, 9) <= HOLD for n in sizes for r in (6, 7) if r in meshes)
    improves = all(mean(n, 6) <= 0.5 * NEAREST_R6[n] for n in sizes if 6 in meshes and n in NEAREST_R6)
    print(f"\nOutcome: {'A2 REPAIRS THE TRANSFER' if repairs else 'IMPROVES' if improves else 'NO EFFECT'}")
    for n in sizes:
        if n in PER_NODE:
            print(f"Refine-4 parity at n_train={n}: {mean(n, 4):.3f} against the per-node encoder's {PER_NODE[n]:.3f} "
                  f"-> {'PARITY' if mean(n, 4) <= PARITY * PER_NODE[n] else 'LOSES ACCURACY AT REFINE 4'}")
    write(rows, a.out, "study6a")


if __name__ == "__main__":
    main()
