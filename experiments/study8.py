"""Study 8: do the Study 7 gains (12 bumps, 1500 steps) survive the checks?

Stage A: certificate and consistency at refine 4 (as Study 4a) for the 12-bump encoder, the FNO and a parameter-matched small FNO.
Stage B: transfer to finer meshes (as Study 5a). Protocol and decision rules: README "Study 8 protocol". Usage:
    OMP_NUM_THREADS=8 uv run python -u -m experiments.study8 [--repeats 3] [--sizes 100,400]
Stage C (cost) is `experiments.study4b --bumps 12 --steps 1500`, run afterwards on a quiet machine.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from baselines import FNO, EncoderWhitney
from compat_galerkin import fem, metrics
from compat_galerkin.device import get_device
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.study4a import BAND, CONSIST, MIN_MATCHED, collect, selected_config
from experiments.study5a import FNOOnMesh
from experiments.study5c import EncoderOnMesh
from experiments.study6a import cached_dataset
from experiments.verification import BOX, analyse, verdict

HOLD, LEAD = 1.25, 0.10


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
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes, meshes = [int(x) for x in a.sizes.split(",")], [int(x) for x in a.meshes.split(",")]
    width, _ = selected_config()
    device = get_device("auto")
    print(f"device {device}; encoder width {width}, 12 bumps, {a.steps} steps", flush=True)

    fams = {r: Family(FamilyConfig(refine=r, grid=64)) for r in meshes}
    fam4 = fams[4]
    MR = fam4.base.interior(fem.region_mass(fam4.base, BOX))
    tests = {r: cached_dataset(fams[r], r, a.ntransfer, 500) for r in meshes}

    s1_rows, cons, matched, s2_rows, counts = [], [], [], [], {}
    for rep in range(a.repeats):
        t0 = time.time()
        train_all = fam4.dataset(max(sizes), seed=400 + rep)
        cal, test = fam4.dataset(a.ncal, seed=600 + rep), fam4.dataset(a.ntest, seed=700 + rep)
        models, fam_models = {}, {}
        for n in sizes:
            train = Dataset(train_all.instances[:n], train_all.e_ref[:n])
            enc = EncoderWhitney(n=12, width=width, seed=rep, schedule=[("galerkin", a.steps)])
            info = enc.fit(fam4, train)
            counts["g12"] = info["params"]
            fno = FNO(epochs=a.epochs, batch=16, seed=rep)
            counts["fno"] = fno.fit(fam4, train, device=device)["params"]
            small = FNO(epochs=a.epochs, batch=16, seed=rep, width=8, modes=6, layers=4)
            counts["fnosmall"] = small.fit(fam4, train, device=device)["params"]
            models[f"g12_n{n}"] = EncoderOnMesh(enc, enc.p0.coords, fam4.base.mesh, "nearest")
            models[f"fno_n{n}"] = FNOOnMesh(fno, device)
            models[f"fnosmall_n{n}"] = FNOOnMesh(small, device)
        print(f"[{rep}] trained {len(models)} models ({time.time() - t0:.0f}s); parameters {counts}", flush=True)

        # Stage A
        rc, _, _, _ = collect(fam4, cal, models, MR)
        rt, fom_true, ratio, tpred = collect(fam4, test, models, MR)
        for s in models:
            s1_rows.append({"repeat": rep, "surrogate": s, "predict_ms": tpred[s], **analyse(rc[s], rt[s], fom_true, ratio)})
            cons.append({"repeat": rep, "surrogate": s, **{f"median_{k}": float(np.median(rt[s][k])) for k in CONSIST}})
        for n in sizes:
            e, f = rt[f"g12_n{n}"], rt[f"fno_n{n}"]
            m = (e["rel_err"] >= BAND[0]) & (e["rel_err"] <= BAND[1]) & (f["rel_err"] >= BAND[0]) & (f["rel_err"] <= BAND[1])
            matched.append({"repeat": rep, "n_train": n, "count": int(m.sum()),
                            "enc_gauss": float(np.median(e["gauss_fine"][m])) if m.sum() >= MIN_MATCHED else float("nan"),
                            "fno_gauss": float(np.median(f["gauss_fine"][m])) if m.sum() >= MIN_MATCHED else float("nan")})
        print(f"  [{rep}] stage A done ({time.time() - t0:.0f}s)", flush=True)

        # Stage B
        for r in meshes:
            fam, ds = fams[r], tests[r]
            errs = {s: [] for s in models}
            for inst, e_ref in zip(ds.instances, ds.e_ref):
                cav = fam.cavity(inst)
                for s, m in models.items():
                    errs[s].append(metrics.evaluate(cav, inst, m.predict(fam, inst, cav), e_ref)["rel_err"])
            for s, e in errs.items():
                s2_rows.append({"repeat": rep, "model": s, "refine": r, "median_field": float(np.median(e))})
            print(f"  [{rep}] refine {r}: " + ", ".join(f"{s} {np.median(e):.3f}" for s, e in errs.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        write(s2_rows, a.out, "study8_transfer_partial")

    mean = lambda rows, key, **kw: float(np.mean([x[key] for x in rows if all(x[k] == v for k, v in kw.items())]))
    print(f"\nParameters: {counts}")
    print("\nStage A, refine 4, medians over the test set (mean over repeats)")
    print(f"{'model':<14}" + "".join(f"{k:>16}" for k in CONSIST))
    for s in dict.fromkeys(c["surrogate"] for c in cons):
        print(f"{s:<14}" + "".join(f"{mean(cons, 'median_' + k, surrogate=s):>16.3e}" for k in CONSIST))

    print("\nParameter-matched comparison (G12 median field error over the small FNO's; LEADS if at most 0.90 at both sizes, TRAILS if at least 1.10 at either)")
    r_small = [mean(cons, "median_rel_err", surrogate=f"g12_n{n}") / mean(cons, "median_rel_err", surrogate=f"fnosmall_n{n}") for n in sizes]
    r_big = [mean(cons, "median_rel_err", surrogate=f"g12_n{n}") / mean(cons, "median_rel_err", surrogate=f"fno_n{n}") for n in sizes]
    outcome = "LEADS" if all(x <= 1 - LEAD for x in r_small) else "TRAILS" if any(x >= 1 + LEAD for x in r_small) else "LEVEL"
    print(f"  against the small FNO ({counts['fnosmall']} parameters): {outcome} (ratios {', '.join(f'{n}: {x:.2f}' for n, x in zip(sizes, r_small))})")
    print(f"  against the large FNO ({counts['fno']} parameters), reported: ratios {', '.join(f'{n}: {x:.2f}' for n, x in zip(sizes, r_big))}")

    vd = verdict(s1_rows)
    print("\nCertificate verdicts")
    for s, v in vd.items():
        print(f"  {s}: {v}")
    print("\nAcceptance / false-accept at alpha = 0.1, tau = 0.10 and 0.20")
    for s in dict.fromkeys(x["surrogate"] for x in s1_rows):
        print(f"  {s:<14}" + "  ".join(f"tau {t}: {mean(s1_rows, f'accept_a0.1_t{t}', surrogate=s):.3f}/{mean(s1_rows, f'falseaccept_a0.1_t{t}', surrogate=s):.3f}" for t in (0.1, 0.2)))

    by_construction, substantive = [], []
    for n in sizes:
        by_construction.append(all(mean(cons, f"median_{k}", surrogate=f"fno_n{n}") >= 100 * max(mean(cons, f"median_{k}", surrogate=f"g12_n{n}"), 1e-16)
                                   for k in ("power_balance", "pec_violation")))
        ok = [m for m in matched if m["n_train"] == n and np.isfinite(m["enc_gauss"])]
        substantive.append(bool(ok) and np.mean([m["enc_gauss"] for m in ok]) <= 0.5 * np.mean([m["fno_gauss"] for m in ok]))
        print(f"n_train={n}: matched instances per repeat {[m['count'] for m in matched if m['n_train'] == n]}; median gauss_fine G12/FNO "
              f"{[(round(m['enc_gauss'], 4), round(m['fno_gauss'], 4)) for m in ok]}")
    cv = "SUBSTANTIVE ADVANTAGE" if all(substantive) else "BY CONSTRUCTION ONLY" if all(by_construction) else "NO ADVANTAGE"
    print(f"Consistency verdict (G12 against the large FNO): {cv}")

    print("\nStage B, median field error by mesh (mean over repeats) and ratio to refine 4")
    for s in dict.fromkeys(x["model"] for x in s2_rows):
        vals = {r: mean(s2_rows, "median_field", model=s, refine=r) for r in meshes}
        held = [r for r in meshes if r != 4 and vals[r] <= HOLD * vals[4]]
        top = 4
        for r in [x for x in meshes if x != 4]:
            if vals[r] <= HOLD * vals[4]:
                top = r
            else:
                break
        print(f"  {s:<14}" + "  ".join(f"r{r}: {vals[r]:.3f}" + (f" ({vals[r] / vals[4]:.2f})" if r != 4 else "") for r in meshes) + f" | holds to refine {top}")
    write(s1_rows, a.out, "study8_certificate")
    write(cons, a.out, "study8_consistency")
    write(s2_rows, a.out, "study8_transfer")
    write(matched, a.out, "study8_matched")


if __name__ == "__main__":
    main()
