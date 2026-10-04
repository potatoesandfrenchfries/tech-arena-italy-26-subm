"""Study 5a: accuracy of the encoder and the FNO, trained at refine 4, on finer meshes (no retraining).

Protocol and decision rules are in the README ("Study 5a protocol"). Needs the Gate 2 selection CSV and the GPU.
    OMP_NUM_THREADS=8 uv run python -m experiments.study5a [--repeats 3] [--meshes 4,5,6,7] [--ntest 200]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from baselines import FNO, CoarseWhitney, EncoderWhitney
from compat_galerkin import metrics
from compat_galerkin.device import get_device
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.study4a import selected_config
from experiments.study4b import Runner
from experiments.study5b import fast_inputs

HOLD = 1.25


class EncoderOnMesh:
    """The trained encoder applied to any mesh (base logits transferred by nearest node)."""

    def __init__(self, model, train_coords):
        self.model, self.coords, self.runners = model, train_coords, {}

    def predict(self, fam, inst, cav):
        key = id(fam)
        if key not in self.runners:
            self.runners[key] = Runner(self.model, fam, self.coords)
        e = np.zeros(cav.n_edges, dtype=complex)
        e[cav.interior_edges] = self.runners[key].predict(inst, cav)
        return e


class FNOOnMesh:
    def __init__(self, model, device):
        self.model, self.device, self.elem = model, device, {}

    @torch.no_grad()
    def predict(self, fam, inst, cav):
        key = id(fam)
        if key not in self.elem:
            g = fam.grid
            self.elem[key] = fam.base.mesh.element_finder()(g.X.ravel(), g.Y.ravel())
        x = fast_inputs(fam, self.elem[key], inst, self.model.mu, self.model.sd, fam.cfg.src_sigma).to(self.device)
        y = self.model.net.eval()(x)[0].cpu().numpy()
        return fam.grid.grid_to_e(np.stack([y[0] + 1j * y[1], y[2] + 1j * y[3]]))


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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes, meshes = [int(x) for x in a.sizes.split(",")], [int(x) for x in a.meshes.split(",")]
    width, steps = selected_config()
    device = get_device("auto")
    print(f"device {device}; encoder width {width}, steps {steps}", flush=True)

    fams, tests = {}, {}
    for r in meshes:
        t0 = time.time()
        fams[r] = Family(FamilyConfig(refine=r, grid=64))
        tests[r] = fams[r].dataset(a.ntest, seed=500)
        print(f"refine {r}: {len(fams[r].base.interior_edges)} unknowns, {a.ntest} test instances with references ({time.time() - t0:.0f}s)", flush=True)
    coarse = {}
    for r in meshes:
        for lvl in (2, 3):
            m = CoarseWhitney(lvl)
            m.fit(fams[r], None)
            coarse[(f"coarse_r{lvl}", r)] = m

    rows = []
    for rep in range(a.repeats):
        t0 = time.time()
        fam4 = fams[4]
        train_all = fam4.dataset(max(sizes), seed=400 + rep)
        models = {}
        for n in sizes:
            train = Dataset(train_all.instances[:n], train_all.e_ref[:n])
            enc = EncoderWhitney(n=8, steps=steps, width=width, seed=rep)
            enc.fit(fam4, train)
            models[f"encoder_n{n}"] = EncoderOnMesh(enc, enc.p0.coords)
            fno = FNO(epochs=a.epochs, batch=16, seed=rep)
            fno.fit(fam4, train, device=device)
            models[f"fno_n{n}"] = FNOOnMesh(fno, device)
        print(f"[{rep}] trained {len(models)} models ({time.time() - t0:.0f}s)", flush=True)
        for r in meshes:
            fam, ds = fams[r], tests[r]
            names = list(models) + ([k[0] for k in coarse if k[1] == r] if rep == 0 else [])
            errs = {s: [] for s in names}
            for inst, e_ref in zip(ds.instances, ds.e_ref):
                cav = fam.cavity(inst)
                for s in names:
                    m = models[s] if s in models else coarse[(s, r)]
                    ev = metrics.evaluate(cav, inst, m.predict(fam, inst, cav), e_ref)
                    errs[s].append((ev["rel_err"], ev["energy_err"]))
            for s in names:
                arr = np.array(errs[s])
                rows.append({"repeat": rep, "model": s, "refine": r, "unknowns": len(fam.base.interior_edges),
                             "median_field": float(np.median(arr[:, 0])), "median_energy": float(np.median(arr[:, 1])),
                             "frac_field10": float(np.mean(arr[:, 0] < 0.10))})
            print(f"  [{rep}] refine {r}: " + ", ".join(
                f"{s} {np.median(np.array(errs[s])[:, 0]):.3f}" for s in names) + f" ({time.time() - t0:.0f}s)", flush=True)
        write(rows, a.out, "study5a_partial")

    mean = lambda model, r, key="median_field": float(np.mean([x[key] for x in rows if x["model"] == model and x["refine"] == r]))
    print("\nMedian field error (mean over repeats) by mesh")
    for s in [f"{k}_n{n}" for n in sizes for k in ("encoder", "fno")] + sorted({k[0] for k in coarse}):
        print(f"  {s:<12}" + "  ".join(f"refine {r}: {mean(s, r):.3f}" for r in meshes))
    print(f"\nRatio to the refine-4 error (hold if at most {HOLD} for the encoder)")
    verdicts = {}
    for n in sizes:
        for k in ("encoder", "fno"):
            s = f"{k}_n{n}"
            ratios = {r: mean(s, r) / mean(s, 4) for r in meshes if r != 4}
            print(f"  {s:<12}" + "  ".join(f"refine {r}: {v:.2f}" for r, v in ratios.items()))
            if k == "encoder":
                held = [r for r in meshes if r != 4]
                top = 4
                for r in held:
                    if ratios[r] <= HOLD:
                        top = r
                    else:
                        break
                verdicts[n] = (top, all(ratios.get(r, 9) <= HOLD for r in (6, 7) if r in meshes))
    for n, (top, speed_ok) in verdicts.items():
        print(f"encoder trained on {n}: TRANSFER HOLDS TO REFINE {top}; accuracy holds where the cost win is (refine 6 and 7): {'YES' if speed_ok else 'NO'}")
    write(rows, a.out, "study5a")


if __name__ == "__main__":
    main()
