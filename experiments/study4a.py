"""Study 4a: consistency at matched accuracy and certificate validity for the encoder, the FNO and POD-136.

Protocol and decision rules are in the README ("Study 4a protocol"). Needs the Gate 2 selection CSV and the GPU
for the FNO. Usage:
    uv run python -m experiments.study4a [--repeats 3] [--sizes 100,400]
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np

from baselines import FNO, EncoderWhitney, PODGalerkin
from compat_galerkin import certify, fem, metrics
from compat_galerkin.device import get_device
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.verification import BOX, analyse, verdict

BAND, MIN_MATCHED = (0.10, 0.30), 100
CONSIST = ("rel_err", "power_balance", "pec_violation", "gauss_fine", "residual")


def selected_config():
    """(width, steps) with the lowest mean validation median field error in the latest Gate 2 selection CSV."""
    rows = list(csv.DictReader(open(sorted(glob.glob("results/gate2_selection_*.csv"))[-1])))
    score = {}
    for r in rows:
        score.setdefault((int(r["width"]), int(r["steps"])), []).append(float(r["val_median_field"]))
    return min(score, key=lambda k: np.mean(score[k]))


def collect(fam, ds, models, MR, time_n=30):
    """Per-instance certificate quantities and consistency metrics for every model; also timing of predict."""
    n, idx = len(ds), fam.base.interior_edges
    names = list(models)
    keys = ("eps", "eta_D", "eta_M", "eta_2", "fom") + CONSIST
    rec = {s: {k: np.zeros(n) for k in keys} for s in names}
    fom_true, ratio = np.zeros(n), np.zeros(n)
    tim = {s: [] for s in names}
    for i, inst in enumerate(ds.instances):
        cav = fam.cavity(inst)
        M = cav.interior(cav.M)
        e_ref_full = ds.e_ref[i]
        e_ref = e_ref_full[idx]
        nref = np.sqrt(np.real(np.vdot(e_ref, M @ e_ref)))
        res = certify.Residual(cav, inst)
        fom_true[i], ratio[i] = np.real(np.vdot(e_ref, MR @ e_ref)), inst.params["ratio"]
        for s in names:
            t0 = time.perf_counter()
            e_full = models[s].predict(fam, inst, cav)
            if i < time_n:
                tim[s].append(time.perf_counter() - t0)
            e = e_full[idx]
            d = e - e_ref
            rec[s]["eps"][i] = np.sqrt(np.real(np.vdot(d, M @ d))) / nref
            ind = res.indicators(e)
            rec[s]["eta_D"][i], rec[s]["eta_M"][i], rec[s]["eta_2"][i] = ind["D"], ind["M"], ind["2"]
            rec[s]["fom"][i] = np.real(np.vdot(e, MR @ e))
            ev = metrics.evaluate(cav, inst, e_full, e_ref_full)
            for k in CONSIST:
                rec[s][k][i] = ev[k]
    return rec, fom_true, ratio, {s: 1e3 * float(np.median(v)) for s, v in tim.items()}


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
    ap.add_argument("--ncal", type=int, default=1000)
    ap.add_argument("--ntest", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    width, steps = selected_config()
    device = get_device("auto")
    print(f"device {device}; encoder width {width}, steps {steps}", flush=True)

    fam = Family(FamilyConfig(refine=4, grid=64))
    MR = fam.base.interior(fem.region_mass(fam.base, BOX))
    rows, cons, matched = [], [], []
    for r in range(a.repeats):
        t0 = time.time()
        train_all = fam.dataset(max(sizes), seed=400 + r)
        cal, test = fam.dataset(a.ncal, seed=600 + r), fam.dataset(a.ntest, seed=700 + r)
        models, fit_s = {}, {}
        for n in sizes:
            train = Dataset(train_all.instances[:n], train_all.e_ref[:n])
            for name, m in (("encoder", EncoderWhitney(n=8, steps=steps, width=width, seed=r)),
                            ("fno", FNO(epochs=a.epochs, batch=16, seed=r)), ("pod136", PODGalerkin(rank=136))):
                info = m.fit(fam, train, device=device)
                models[f"{name}_n{n}"], fit_s[f"{name}_n{n}"] = m, info.get("fit_seconds", float("nan"))
        print(f"[{r}] trained {len(models)} models ({time.time() - t0:.0f}s)", flush=True)
        rc, _, _, _ = collect(fam, cal, models, MR)
        rt, fom_true, ratio, tpred = collect(fam, test, models, MR)
        for s in models:
            rows.append({"repeat": r, "surrogate": s, "predict_ms": tpred[s], "fit_seconds": fit_s[s], **analyse(rc[s], rt[s], fom_true, ratio)})
            cons.append({"repeat": r, "surrogate": s, **{f"median_{k}": float(np.median(rt[s][k])) for k in CONSIST}})
        for n in sizes:  # matched accuracy: instances where the encoder and the FNO are both inside the band
            e, f = rt[f"encoder_n{n}"], rt[f"fno_n{n}"]
            m = (e["rel_err"] >= BAND[0]) & (e["rel_err"] <= BAND[1]) & (f["rel_err"] >= BAND[0]) & (f["rel_err"] <= BAND[1])
            matched.append({"repeat": r, "n_train": n, "count": int(m.sum()),
                            "enc_gauss": float(np.median(e["gauss_fine"][m])) if m.sum() >= MIN_MATCHED else float("nan"),
                            "fno_gauss": float(np.median(f["gauss_fine"][m])) if m.sum() >= MIN_MATCHED else float("nan")})
        print(f"repeat {r}: {time.time() - t0:.0f}s", flush=True)

    mean = lambda rs, k: float(np.mean([x[k] for x in rs]))
    print("\nConsistency, median over the test set (mean over repeats)")
    print(f"{'model':<14}" + "".join(f"{k:>16}" for k in CONSIST) + f"{'predict ms':>12}")
    for s in dict.fromkeys(c["surrogate"] for c in cons):
        cs = [c for c in cons if c["surrogate"] == s]
        print(f"{s:<14}" + "".join(f"{mean(cs, 'median_' + k):>16.3e}" for k in CONSIST)
              + f"{mean([x for x in rows if x['surrogate'] == s], 'predict_ms'):>12.1f}")
    by_construction, substantive = [], []
    for n in sizes:
        cs = lambda name: [c for c in cons if c["surrogate"] == f"{name}_n{n}"]
        eps_floor = 1e-16
        by_construction.append(all(mean(cs("fno"), f"median_{k}") >= 100 * max(mean(cs("encoder"), f"median_{k}"), eps_floor)
                                   for k in ("power_balance", "pec_violation")))
        ms = [m for m in matched if m["n_train"] == n]
        ok = [m for m in ms if np.isfinite(m["enc_gauss"])]
        substantive.append(bool(ok) and np.mean([m["enc_gauss"] for m in ok]) <= 0.5 * np.mean([m["fno_gauss"] for m in ok]))
        print(f"n_train={n}: matched instances per repeat {[m['count'] for m in ms]}; median gauss_fine encoder/FNO "
              f"{[(round(m['enc_gauss'], 4), round(m['fno_gauss'], 4)) for m in ok]}")
    if all(substantive):
        cv = "SUBSTANTIVE ADVANTAGE"
    elif all(by_construction):
        cv = "BY CONSTRUCTION ONLY"
    else:
        cv = "NO ADVANTAGE"
    print(f"Consistency verdict: {cv} (by construction at each size: {by_construction}; substantive: {substantive})")

    print("\nCertificate verdicts")
    for s, v in verdict(rows).items():
        print(f"  {s}: {v}")
    print("\nAcceptance rate / false-accept rate at alpha = 0.1, tau = 0.05, 0.10, 0.20 (mean over repeats)")
    for s in dict.fromkeys(x["surrogate"] for x in rows):
        rs = [x for x in rows if x["surrogate"] == s]
        print(f"  {s:<14}" + "  ".join(f"tau {t}: {mean(rs, f'accept_a0.1_t{t}'):.3f}/{mean(rs, f'falseaccept_a0.1_t{t}'):.3f}" for t in (0.05, 0.1, 0.2)))

    write(rows, a.out, "study4a_certificate")
    write(cons, a.out, "study4a_consistency")
    write(matched, a.out, "study4a_matched")


if __name__ == "__main__":
    main()
