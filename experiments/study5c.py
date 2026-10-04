"""Study 5c (variant A1): is the transfer failure of Study 5a caused by the nearest-node transfer of the base logits?

The same encoder, trained at refine 4, is applied to finer meshes with two transfers of its shared base logits: nearest node (N)
and linear P1 interpolation (L). Protocol and decision rules: README "Study 5c protocol". Usage:
    OMP_NUM_THREADS=8 uv run python -m experiments.study5c [--repeats 3] [--meshes 4,5,6,7] [--ntest 200]
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np
import torch

from baselines import EncoderWhitney
from compat_galerkin import lift, metrics
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.study4a import selected_config
from experiments.study4b import Runner

HOLD = 1.25
VARIANTS = ("nearest", "linear")


class EncoderOnMesh:
    def __init__(self, model, train_coords, train_mesh, transfer):
        self.model, self.coords, self.mesh, self.transfer, self.runners = model, train_coords, train_mesh, transfer, {}

    def runner(self, fam):
        key = id(fam)
        if key not in self.runners:
            self.runners[key] = Runner(self.model, fam, self.coords, self.transfer, self.mesh)
        return self.runners[key]

    def predict(self, fam, inst, cav):
        e = np.zeros(cav.n_edges, dtype=complex)
        e[cav.interior_edges] = self.runner(fam).predict(inst, cav)
        return e


@torch.no_grad()
def roughness(model, fam, inst):
    """Mean absolute edge difference of the partition of unity (mean over interior edges and bumps of |G W|)."""
    r = model.runner(fam)
    W = lift.make_pou(r.logits(r.inputs(inst))[0], r.mask)[:, : model.model.n]
    return float(torch.sparse.mm(r.G, W)[r.idx].abs().mean())


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
    print(f"encoder width {width}, steps {steps}", flush=True)

    fams, tests = {}, {}
    for r in meshes:
        t0 = time.time()
        fams[r] = Family(FamilyConfig(refine=r, grid=64))
        tests[r] = fams[r].dataset(a.ntest, seed=500)
        print(f"refine {r}: {len(fams[r].base.interior_edges)} unknowns, {a.ntest} test instances with references ({time.time() - t0:.0f}s)", flush=True)

    Path(a.out, "models").mkdir(parents=True, exist_ok=True)
    rows, rough = [], []
    for rep in range(a.repeats):
        t0 = time.time()
        fam4 = fams[4]
        train_all = fam4.dataset(max(sizes), seed=400 + rep)
        models = {}
        for n in sizes:
            enc = EncoderWhitney(n=8, steps=steps, width=width, seed=rep)
            enc.fit(fam4, Dataset(train_all.instances[:n], train_all.e_ref[:n]))
            torch.save({"net": enc.net.state_dict(), "base": enc.base, "mu": enc.mu, "sd": enc.sd, "n": 8, "width": width,
                        "steps": steps, "n_train": n, "repeat": rep, "train_seed": 400 + rep},
                       Path(a.out) / "models" / f"encoder_n{n}_rep{rep}.pt")
            for v in VARIANTS:
                models[(n, v)] = EncoderOnMesh(enc, enc.p0.coords, fam4.base.mesh, v)
        print(f"[{rep}] trained {len(sizes)} encoders ({time.time() - t0:.0f}s)", flush=True)
        for r in meshes:
            fam, ds = fams[r], tests[r]
            errs = {k: [] for k in models}
            for inst, e_ref in zip(ds.instances, ds.e_ref):
                cav = fam.cavity(inst)
                for k, m in models.items():
                    ev = metrics.evaluate(cav, inst, m.predict(fam, inst, cav), e_ref)
                    errs[k].append((ev["rel_err"], ev["energy_err"]))
            for (n, v), e in errs.items():
                arr = np.array(e)
                rows.append({"repeat": rep, "n_train": n, "variant": v, "refine": r, "median_field": float(np.median(arr[:, 0])),
                             "median_energy": float(np.median(arr[:, 1]))})
            if rep == 0 and r != 4:
                for v in VARIANTS:
                    rough.append({"refine": r, "variant": v, "mean_abs_edge_difference": roughness(models[(sizes[0], v)], fam, ds.instances[0])})
            print(f"  [{rep}] refine {r}: " + ", ".join(f"n={n} {v} {np.median(np.array(e)[:, 0]):.3f}" for (n, v), e in errs.items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
        write(rows, a.out, "study5c_partial")

    mean = lambda n, v, r: float(np.mean([x["median_field"] for x in rows if x["n_train"] == n and x["variant"] == v and x["refine"] == r]))
    print("\nMedian field error (mean over repeats): variant N = nearest node, L = linear")
    for n in sizes:
        for v in VARIANTS:
            print(f"  n_train={n} {v:<8}" + "  ".join(f"refine {r}: {mean(n, v, r):.3f}" for r in meshes))
    prev = sorted(glob.glob(str(Path(a.out) / "study5a_2*.csv")))
    if prev:
        old = list(csv.DictReader(open(prev[-1])))
        print("Sanity, Study 5a encoder (nearest node), mean median field error: " + "; ".join(
            f"n={n}: " + ", ".join(f"refine {r}: {np.mean([float(x['median_field']) for x in old if x['model'] == f'encoder_n{n}' and int(x['refine']) == r]):.3f}"
                                   for r in meshes) for n in sizes))
    print("\nRatio to the refine-4 error (hold if at most 1.25)")
    holds = {}
    for n in sizes:
        for v in VARIANTS:
            ratios = {r: mean(n, v, r) / mean(n, v, 4) for r in meshes if r != 4}
            holds[(n, v)] = ratios
            print(f"  n_train={n} {v:<8}" + "  ".join(f"refine {r}: {x:.2f}" for r, x in ratios.items()))
    repairs = all(holds[(n, "linear")].get(r, 9) <= HOLD for n in sizes for r in (6, 7) if r in meshes)
    partial = all(mean(n, "linear", 6) <= 0.5 * mean(n, "nearest", 6) for n in sizes if 6 in meshes)
    print(f"\nOutcome: {'A1 REPAIRS THE TRANSFER' if repairs else 'PARTIAL' if partial else 'NO EFFECT'}")
    print("Roughness of the partition of unity (mean |G W| over interior edges and bumps), repeat 0, 100 instances:")
    for x in rough:
        print(f"  refine {x['refine']} {x['variant']:<8} {x['mean_abs_edge_difference']:.4f}")
    write(rows, a.out, "study5c")
    write(rough, a.out, "study5c_roughness")


if __name__ == "__main__":
    main()
