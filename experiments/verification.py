"""Verification study: residual indicators, conformal certificate, selection test.

Protocol and decision rules are in the README ("Verification study protocol"). Usage:
    uv run python -m experiments.verification [--repeats 5] [--ncal 1000] [--ntest 1000] [--ntrain 300]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from baselines import CoarseWhitney, PODGalerkin
from compat_galerkin import certify, fem
from compat_galerkin.problems import Family, FamilyConfig

TAUS = (0.02, 0.05, 0.10, 0.20)
ALPHAS = (0.1, 0.05)
BOX = (0.65, 0.85, 0.15, 0.35)  # target region R of the figure of merit
SEL_FRAC, TOPK, TAU_TOPK = 0.10, 10, 0.10
BINS = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
COV_TOL = 0.02
SPEARMAN_TRACK, SPEARMAN_FAIL = 0.9, 0.7


def make_surrogates(fam, train):
    out = {}
    for r in (2, 3):
        m = CoarseWhitney(r)
        m.fit(fam, None)
        out[f"coarse_r{r}"] = m
    for rank in (16, 64, 128):
        m = PODGalerkin(rank)
        m.fit(fam, train)
        out[f"pod_{rank}"] = m
    return out


def collect(fam, ds, surrogates, MR):
    """Per-instance true error, indicators and figure of merit for every surrogate."""
    n = len(ds)
    names = list(surrogates)
    rec = {s: {k: np.zeros(n) for k in ("eps", "eta_D", "eta_M", "eta_2", "fom")} for s in names}
    fom_true, ratio = np.zeros(n), np.zeros(n)
    idx = fam.base.interior_edges
    for i, inst in enumerate(ds.instances):
        cav = fam.cavity(inst)
        M = cav.interior(cav.M)
        e_ref = ds.e_ref[i][idx]
        nref = np.sqrt(np.real(np.vdot(e_ref, M @ e_ref)))
        res = certify.Residual(cav, inst)
        fom_true[i] = np.real(np.vdot(e_ref, MR @ e_ref))
        ratio[i] = inst.params["ratio"]
        for s in names:
            e = surrogates[s].predict(fam, inst, cav)[idx]
            d = e - e_ref
            rec[s]["eps"][i] = np.sqrt(np.real(np.vdot(d, M @ d))) / nref
            ind = res.indicators(e)
            rec[s]["eta_D"][i], rec[s]["eta_M"][i], rec[s]["eta_2"][i] = ind["D"], ind["M"], ind["2"]
            rec[s]["fom"][i] = np.real(np.vdot(e, MR @ e))
    return rec, fom_true, ratio


def analyse(cal, test, fom_true_test, ratio_test):
    """All scalar metrics for one surrogate on one repeat."""
    out = {"eps_median": float(np.median(test["eps"]))}
    for ind in ("D", "M", "2"):
        eta_c, eta_t = cal[f"eta_{ind}"], test[f"eta_{ind}"]
        out[f"spearman_{ind}"] = float(spearmanr(np.log(eta_t), np.log(test["eps"])).correlation)
        out[f"cov_{ind}_a0.1"] = certify.certificate(cal["eps"], eta_c, test["eps"], eta_t, 0.1)["coverage"]
    out["cov_naive_D"] = float(np.mean(test["eps"] <= test["eta_D"]))
    for a in ALPHAS:
        c = certify.certificate(cal["eps"], cal["eta_D"], test["eps"], test["eta_D"], a, TAUS)
        out[f"qhat_a{a}"], out[f"cov_a{a}"] = c["q_hat"], c["coverage"]
        for t in TAUS:
            out[f"accept_a{a}_t{t}"], out[f"falseaccept_a{a}_t{t}"] = c[f"accept_{t}"], c[f"false_accept_{t}"]

    # Selection test: top SEL_FRAC of the test set by surrogate figure of merit.
    q = certify.conformal_quantile(cal["eps"] / cal["eta_D"], 0.1)
    B = q * test["eta_D"]
    nsel = max(1, int(SEL_FRAC * len(B)))
    sel = np.argsort(-test["fom"])[:nsel]
    out["cov_sel_a0.1"] = float(np.mean(test["eps"][sel] <= B[sel]))
    out["falseaccept_sel_t0.1"] = float(np.mean((B[sel] <= 0.1) & (test["eps"][sel] > 0.1)))
    out["accept_sel_t0.1"] = float(np.mean(B[sel] <= 0.1))
    bias = (test["fom"] - fom_true_test) / fom_true_test
    out["fom_bias_all"], out["fom_bias_sel"] = float(np.mean(bias)), float(np.mean(bias[sel]))
    sel_cal = np.argsort(-cal["fom"])[: max(1, int(SEL_FRAC * len(cal["fom"])))]
    q_sel = certify.conformal_quantile(cal["eps"][sel_cal] / cal["eta_D"][sel_cal], 0.1)
    out["cov_sel_recal_a0.1"] = float(np.mean(test["eps"][sel] <= q_sel * test["eta_D"][sel]))

    # Verified top-K: the K best by surrogate FoM, certified at tau = TAU_TOPK or solved by full-wave.
    top = np.argsort(-test["fom"])[:TOPK]
    out["topk_calls"] = int(np.sum(B[top] > TAU_TOPK))
    out["topk_true_fom_ratio"] = float(np.mean(fom_true_test[top]) / np.mean(np.sort(fom_true_test)[-TOPK:]))

    for lo, hi in BINS:
        m = (ratio_test >= lo) & (ratio_test < hi + (1e-12 if hi >= 0.6 else 0))
        out[f"cov_bin_{lo}_{hi}"] = float(np.mean(test["eps"][m] <= B[m])) if m.any() else float("nan")
    return out


def timing(fam, ds, surrogates, n=100):
    idx = fam.base.interior_edges
    t_full, t_res, t_sur = [], [], {s: [] for s in surrogates}
    for inst in ds.instances[:n]:
        cav = fam.cavity(inst)
        t0 = time.perf_counter()
        e = fem.solve_fine(cav, inst.omega2, inst.f)
        t_full.append(time.perf_counter() - t0)
        res = certify.Residual(cav, inst)
        t0 = time.perf_counter()
        res.eta_D(e[idx])
        t_res.append(time.perf_counter() - t0)
        for s, m in surrogates.items():
            t0 = time.perf_counter()
            m.predict(fam, inst, cav)
            t_sur[s].append(time.perf_counter() - t0)
    return {"full_solve_ms": 1e3 * np.median(t_full), "eta_D_ms": 1e3 * np.median(t_res),
            **{f"predict_{s}_ms": 1e3 * np.median(v) for s, v in t_sur.items()}}


def verdict(rows):
    """Per-surrogate verdict from the means over repeats, by the registered rules."""
    out = {}
    for s in sorted({r["surrogate"] for r in rows}):
        rs = [r for r in rows if r["surrogate"] == s]
        m = lambda k: float(np.mean([r[k] for r in rs]))
        rho, c_rand, c_sel = m("spearman_D"), m("cov_a0.1"), m("cov_sel_a0.1")
        ok = lambda c: abs(c - 0.9) <= COV_TOL + 1e-9
        if rho < SPEARMAN_FAIL:
            v = "DOES NOT TRACK"
        elif not ok(c_rand):
            v = "INCONCLUSIVE (random coverage off)"
        elif not ok(c_sel):
            v = "NEEDS RECALIBRATION"
        elif rho >= SPEARMAN_TRACK:
            v = "CERTIFICATE VALID"
        else:
            v = "INCONCLUSIVE (weak tracking)"
        out[s] = f"{v} (Spearman {rho:.3f}, random coverage {c_rand:.3f}, selected coverage {c_sel:.3f})"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ntrain", type=int, default=300)
    ap.add_argument("--ncal", type=int, default=1000)
    ap.add_argument("--ntest", type=int, default=1000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    fam = Family(FamilyConfig(refine=4))
    MR = fam.base.interior(fem.region_mass(fam.base, BOX))
    rows, tim = [], None
    for r in range(a.repeats):
        t0 = time.time()
        train = fam.dataset(a.ntrain, seed=100 + r)
        cal = fam.dataset(a.ncal, seed=200 + r)
        test = fam.dataset(a.ntest, seed=300 + r)
        sur = make_surrogates(fam, train)
        rc, _, _ = collect(fam, cal, sur, MR)
        rt, fom_true, ratio = collect(fam, test, sur, MR)
        if tim is None:
            tim = timing(fam, test, sur)
        for s in sur:
            rows.append({"repeat": r, "surrogate": s, **analyse(rc[s], rt[s], fom_true, ratio)})
        print(f"repeat {r}: done in {time.time() - t0:.0f}s", flush=True)

    keys = ["eps_median", "spearman_D", "spearman_M", "spearman_2", "cov_naive_D", "cov_a0.1", "cov_a0.05",
            "cov_sel_a0.1", "cov_sel_recal_a0.1", "fom_bias_all", "fom_bias_sel", "accept_a0.1_t0.05",
            "falseaccept_a0.1_t0.05", "accept_a0.1_t0.2", "topk_calls", "topk_true_fom_ratio"]
    print("\nMean (std) over repeats")
    print(f"{'surrogate':<11}" + "".join(f"{k[:17]:>19}" for k in keys[:8]))
    for s in sorted({r["surrogate"] for r in rows}):
        rs = [r for r in rows if r["surrogate"] == s]
        print(f"{s:<11}" + "".join(f"{np.mean([r[k] for r in rs]):>12.3f}({np.std([r[k] for r in rs]):.3f})" for k in keys[:8]))
    print(f"\n{'surrogate':<11}" + "".join(f"{k[:17]:>19}" for k in keys[8:]))
    for s in sorted({r["surrogate"] for r in rows}):
        rs = [r for r in rows if r["surrogate"] == s]
        print(f"{s:<11}" + "".join(f"{np.mean([r[k] for r in rs]):>12.3f}({np.std([r[k] for r in rs]):.3f})" for k in keys[8:]))
    print("\nAcceptance rate / false-accept rate at alpha = 0.1 (mean over repeats)")
    for s in sorted({r["surrogate"] for r in rows}):
        rs = [r for r in rows if r["surrogate"] == s]
        print(f"  {s:<10}" + "  ".join(
            f"tau {t}: {np.mean([r[f'accept_a0.1_t{t}'] for r in rs]):.3f}/{np.mean([r[f'falseaccept_a0.1_t{t}'] for r in rs]):.3f}"
            for t in TAUS))
    print("\nCoverage by r bin (alpha = 0.1, mean over repeats)")
    for s in sorted({r["surrogate"] for r in rows}):
        rs = [r for r in rows if r["surrogate"] == s]
        print(f"  {s:<10}" + "  ".join(f"[{lo},{hi}]: {np.nanmean([r[f'cov_bin_{lo}_{hi}'] for r in rs]):.3f}" for lo, hi in BINS))
    print("\nTiming, median per instance (ms):", {k: round(v, 2) for k, v in tim.items()})
    print("\nVerdicts:")
    for s, v in verdict(rows).items():
        print(f"  {s}: {v}")

    Path(a.out).mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(a.out) / f"verification_{stamp}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(Path(a.out) / f"verification_timing_{stamp}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tim.keys()))
        w.writeheader()
        w.writerow(tim)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
