"""Study 6b (variant C1): the FNO's field added to a coarse Whitney space, then a Galerkin solve.

Stage 1 (refine 4): accuracy, consistency and certificate against the FNO. Stage 2: transfer to finer meshes.
Protocol and decision rules: README "Study 6b protocol". Needs the GPU for the FNO. Usage:
    OMP_NUM_THREADS=8 uv run python -m experiments.study6b [--repeats 3] [--meshes 4,5,6,7]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from baselines import FNO, CoarseWhitney
from baselines.enriched_galerkin import EnrichedGalerkin
from compat_galerkin import fem, metrics
from compat_galerkin.device import get_device
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.study4a import CONSIST, collect
from experiments.study5a import FNOOnMesh
from experiments.study6a import cached_dataset
from experiments.verification import BOX, analyse, verdict

HOLD, RETAIN, IMPROVE = 1.25, 1.10, 0.90
VARIANTS = (2, 3)


class CachedFNO:
    """FNO prediction shared between the FNO entry and the C1 variants (one forward pass per instance)."""

    def __init__(self, model, device):
        self.inner, self.key, self.value = FNOOnMesh(model, device), None, None

    def predict(self, fam, inst, cav):
        key = (id(fam), id(inst))
        if key != self.key:
            self.key, self.value = key, self.inner.predict(fam, inst, cav)
        return self.value


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
    ap.add_argument("--ncal", type=int, default=1000)
    ap.add_argument("--ntest", type=int, default=1000)
    ap.add_argument("--ntransfer", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes, meshes = [int(x) for x in a.sizes.split(",")], [int(x) for x in a.meshes.split(",")]
    device = get_device("auto")
    print(f"device {device}", flush=True)

    fams = {r: Family(FamilyConfig(refine=r, grid=64)) for r in meshes}
    fam4 = fams[4]
    MR = fam4.base.interior(fem.region_mass(fam4.base, BOX))
    coarse = {}
    for r in meshes:
        for lvl in VARIANTS:
            m = CoarseWhitney(lvl)
            m.fit(fams[r], None)
            coarse[(lvl, r)] = m
    tests = {}
    for r in meshes:
        t0 = time.time()
        tests[r] = cached_dataset(fams[r], r, a.ntransfer, 500)
        print(f"refine {r}: {len(fams[r].base.interior_edges)} unknowns, {a.ntransfer} transfer instances ({time.time() - t0:.0f}s)", flush=True)

    s1_rows, cons, s2_rows = [], [], []
    for rep in range(a.repeats):
        t0 = time.time()
        train_all = fam4.dataset(max(sizes), seed=400 + rep)
        cal, test = fam4.dataset(a.ncal, seed=600 + rep), fam4.dataset(a.ntest, seed=700 + rep)
        fnos, models = {}, {}
        for n in sizes:
            fno = FNO(epochs=a.epochs, batch=16, seed=rep)
            fno.fit(fam4, Dataset(train_all.instances[:n], train_all.e_ref[:n]), device=device)
            fnos[n] = CachedFNO(fno, device)
            models[f"fno_n{n}"] = fnos[n]
            for lvl in VARIANTS:
                models[f"c1_r{lvl}_n{n}"] = EnrichedGalerkin(coarse[(lvl, 4)], fnos[n].predict)
        if rep == 0:
            for lvl in VARIANTS:
                models[f"coarse_r{lvl}"] = coarse[(lvl, 4)]
        print(f"[{rep}] trained {len(sizes)} FNOs ({time.time() - t0:.0f}s)", flush=True)

        # Stage 1: refine 4
        rc, _, _, _ = collect(fam4, cal, models, MR)
        rt, fom_true, ratio, tpred = collect(fam4, test, models, MR)
        for s in models:
            s1_rows.append({"repeat": rep, "surrogate": s, "predict_ms": tpred[s], **analyse(rc[s], rt[s], fom_true, ratio)})
            cons.append({"repeat": rep, "surrogate": s, **{f"median_{k}": float(np.median(rt[s][k])) for k in CONSIST}})
        print(f"  [{rep}] stage 1 done ({time.time() - t0:.0f}s)", flush=True)

        # Stage 2: transfer to finer meshes (the FNO and the C1 spaces rebuilt on each mesh)
        for r in meshes:
            fam, ds = fams[r], tests[r]
            names = [f"fno_n{n}" for n in sizes] + [f"c1_r{lvl}_n{n}" for n in sizes for lvl in VARIANTS]
            if rep == 0:
                names += [f"coarse_r{lvl}" for lvl in VARIANTS]
            errs = {s: [] for s in names}
            for inst, e_ref in zip(ds.instances, ds.e_ref):
                cav = fam.cavity(inst)
                for s in names:
                    if s.startswith("fno"):
                        e = fnos[int(s.split("_n")[1])].predict(fam, inst, cav)
                    elif s.startswith("c1"):
                        lvl, n = int(s[4]), int(s.split("_n")[1])
                        e = EnrichedGalerkin(coarse[(lvl, r)], fnos[n].predict).predict(fam, inst, cav)
                    else:
                        e = coarse[(int(s[-1]), r)].predict(fam, inst, cav)
                    errs[s].append(metrics.evaluate(cav, inst, e, e_ref)["rel_err"])
            for s, e in errs.items():
                s2_rows.append({"repeat": rep, "model": s, "refine": r, "median_field": float(np.median(e)),
                                "frac_field10": float(np.mean(np.array(e) < 0.10))})
            print(f"  [{rep}] refine {r}: " + ", ".join(f"{s} {np.median(e):.3f}" for s, e in errs.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        write(s2_rows, a.out, "study6b_transfer_partial")

    mean = lambda rows, key, **kw: float(np.mean([x[key] for x in rows if all(x[k] == v for k, v in kw.items())]))
    print("\nStage 1, refine 4, median over the test set (mean over repeats)")
    print(f"{'model':<14}" + "".join(f"{k:>16}" for k in CONSIST))
    for s in dict.fromkeys(c["surrogate"] for c in cons):
        print(f"{s:<14}" + "".join(f"{mean(cons, 'median_' + k, surrogate=s):>16.3e}" for k in CONSIST))
    vd = verdict(s1_rows)
    print("\nCertificate verdicts")
    for s, v in vd.items():
        print(f"  {s}: {v}")
    print("\nAcceptance / false-accept at alpha = 0.1, tau = 0.10 and 0.20")
    for s in dict.fromkeys(x["surrogate"] for x in s1_rows):
        print(f"  {s:<14}" + "  ".join(f"tau {t}: {mean(s1_rows, f'accept_a0.1_t{t}', surrogate=s):.3f}/{mean(s1_rows, f'falseaccept_a0.1_t{t}', surrogate=s):.3f}" for t in (0.1, 0.2)))

    print("\nStage 2, median field error by mesh (mean over repeats)")
    names2 = list(dict.fromkeys(x["model"] for x in s2_rows))
    for s in names2:
        print(f"  {s:<14}" + "  ".join(f"refine {r}: {mean(s2_rows, 'median_field', model=s, refine=r):.3f}" for r in meshes))

    print("\nCriteria")
    overall = {}
    for lvl in VARIANTS:
        for n in sizes:
            v, f = f"c1_r{lvl}_n{n}", f"fno_n{n}"
            acc = mean(cons, "median_rel_err", surrogate=v) / mean(cons, "median_rel_err", surrogate=f)
            accuracy = "IMPROVED" if acc <= IMPROVE else "RETAINED" if acc <= RETAIN else "DEGRADED"
            pb, res = mean(cons, "median_power_balance", surrogate=v), mean(cons, "median_residual", surrogate=v)
            consistency = pb <= 1e-10 and res <= mean(cons, "median_residual", surrogate=f) / 3
            rho = mean(s1_rows, "spearman_D", surrogate=v)
            ratios = {r: mean(s2_rows, "median_field", model=v, refine=r) / mean(s2_rows, "median_field", model=v, refine=4)
                      for r in meshes if r != 4}
            transfer = all(ratios.get(r, 9) <= HOLD for r in (6, 7) if r in meshes)
            overall[(lvl, n)] = (accuracy, consistency, rho >= 0.9, transfer)
            print(f"  {v}: accuracy {accuracy} ({acc:.2f} x FNO), consistency {'RESTORED' if consistency else 'NOT RESTORED'} "
                  f"(power balance {pb:.1e}, residual {res:.2f}), certificate: {vd[v]}, transfer {'HOLDS' if transfer else 'FAILS'} "
                  f"(ratios {', '.join(f'r{r}: {x:.2f}' for r, x in ratios.items())})")
    works = any(all(overall[(lvl, n)][0] != "DEGRADED" and overall[(lvl, n)][1] and overall[(lvl, n)][2] and overall[(lvl, n)][3] for n in sizes)
                for lvl in VARIANTS)
    partial = any(all(overall[(lvl, n)][0] != "DEGRADED" and overall[(lvl, n)][1] for n in sizes) for lvl in VARIANTS)
    failed = all(any(overall[(lvl, n)][0] == "DEGRADED" for n in sizes) for lvl in VARIANTS)
    print(f"\nOverall: {'C1 WORKS' if works else 'PARTIAL' if partial else 'FAILS' if failed else 'FAILS (accuracy or consistency not met)'}")
    write(s1_rows, a.out, "study6b_certificate")
    write(cons, a.out, "study6b_consistency")
    write(s2_rows, a.out, "study6b_transfer")


if __name__ == "__main__":
    main()
