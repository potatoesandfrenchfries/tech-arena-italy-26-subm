"""Gate 2: does an encoder-predicted compatible basis beat POD at small training sizes?

Stage 1 selects (width, steps) on validation data (seeds 900 and 901); stage 2 trains the selected encoder on the
data-efficiency splits (nested subsets of seed 400 + r, 300 test instances of seed 500 + r) and applies the
registered criteria A and B against the committed baseline numbers. Protocol: README "Gate 2 protocol".
Usage:
    uv run python -m experiments.gate2 [--repeats 3] [--sizes 25,50,100,200,400]
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np

from baselines import EncoderWhitney
from compat_galerkin import fem
from compat_galerkin.problems import Dataset, Family, FamilyConfig
from experiments.gate1b_geometry import FIELD_TOL, evaluate_method, score

GRID = [(w, s) for w in (12, 24) for s in (500, 1000)]
SELECT_SIZES = (25, 100)
ORACLE_MEDIAN = 0.044


def baseline_numbers(pattern="results/data_efficiency_2*.csv"):
    """Mean over repeats of the median field error, by (method, n_train), from the latest baseline CSV."""
    rows = list(csv.DictReader(open(sorted(glob.glob(pattern))[-1])))
    out = {}
    for r in rows:
        out.setdefault((r["method"], int(r["n_train"])), []).append((int(r["repeat"]), float(r["median_field"])))
    return {k: dict(v) for k, v in out.items()}


def select(fam, out):
    val, trn = fam.dataset(100, seed=900), fam.dataset(100, seed=901)
    ratios = np.array([i.params["ratio"] for i in val.instances])
    rows = []
    for width, steps in GRID:
        for n in SELECT_SIZES:
            m = EncoderWhitney(n=8, steps=steps, width=width, seed=0)
            info = m.fit(fam, Dataset(trn.instances[:n], trn.e_ref[:n]))
            s = score(evaluate_method(fam, m, val), ratios)
            rows.append({"width": width, "steps": steps, "n_train": n, "train_error": info["train_error"],
                         "val_median_field": s["median_field"], "val_median_energy": s["median_energy"]})
            print(f"  select width={width:>2} steps={steps:>4} n_train={n:>3}: train {info['train_error']:.3f} | "
                  f"val median field {s['median_field']:.3f} energy {s['median_energy']:.3f} ({info['fit_seconds']:.0f}s)", flush=True)
    scores = {}
    for width, steps in GRID:
        scores[(width, steps)] = np.mean([r["val_median_field"] for r in rows if r["width"] == width and r["steps"] == steps])
    best = min(scores, key=scores.get)
    print("  mean validation median field error by configuration: " + ", ".join(f"{k}: {v:.3f}" for k, v in scores.items()))
    print(f"  selected (width, steps) = {best}", flush=True)
    write(rows, out, "gate2_selection")
    return best


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
    ap.add_argument("--sizes", default="25,50,100,200,400")
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    fam = Family(FamilyConfig(refine=4, grid=64))

    print("Stage 1: hyperparameter selection on validation data", flush=True)
    width, steps = select(fam, a.out)

    print("\nStage 2: test", flush=True)
    base = baseline_numbers()
    rows = []
    for r in range(a.repeats):
        t0 = time.time()
        train_all = fam.dataset(max(sizes), seed=400 + r)
        test = fam.dataset(a.ntest, seed=500 + r)
        ratios = np.array([i.params["ratio"] for i in test.instances])
        for n in sizes:
            m = EncoderWhitney(n=8, steps=steps, width=width, seed=r)
            info = m.fit(fam, Dataset(train_all.instances[:n], train_all.e_ref[:n]))
            s = score(evaluate_method(fam, m, test), ratios)
            rows.append({"repeat": r, "method": "encoder", "n_train": n, "dim": info["dim"], "params": info["params"],
                         "train_error": info["train_error"], "fit_seconds": info["fit_seconds"], "width": width, "steps": steps, **s})
            print(f"  [{r}] n_train={n:>3} encoder: median field {s['median_field']:.3f} mean {s['mean_field']:.3f} | median energy "
                  f"{s['median_energy']:.3f} | energy<5%: {s['frac_energy5']:.2f} field<10%: {s['frac_field10']:.2f} | train {info['train_error']:.3f} "
                  f"({info['fit_seconds']:.0f}s)", flush=True)
        if r == 0:  # cost of one encoder pass (grid inputs, lift, reduced solve) against the fine solve, at the last trained model
            sub = test.instances[:30]
            tp, tf = [], []
            for inst in sub:
                cav = fam.cavity(inst)
                t = time.perf_counter(); m.predict(fam, inst, cav); tp.append(time.perf_counter() - t)
                t = time.perf_counter(); fem.solve_fine(cav, inst.omega2, inst.f); tf.append(time.perf_counter() - t)
            rows.append({"repeat": r, "method": "timing", "predict_ms_median": 1e3 * float(np.median(tp)),
                         "fine_solve_ms_median": 1e3 * float(np.median(tf))})
            print(f"  one encoder predict {1e3 * np.median(tp):.1f} ms against one fine solve {1e3 * np.median(tf):.1f} ms (median of 30)", flush=True)
        print(f"repeat {r}: {time.time() - t0:.0f}s", flush=True)

    enc = {n: float(np.mean([x["median_field"] for x in rows if x["method"] == "encoder" and x["n_train"] == n])) for n in sizes}
    print("\nMean over repeats, median field error / median energy error:")
    for n in sizes:
        e = np.mean([x["median_energy"] for x in rows if x["method"] == "encoder" and x["n_train"] == n])
        cmp = ", ".join(f"{m} {np.mean(list(base[(m, n)].values())):.3f}" for m in ("fno", "pod36", "pod136") if (m, n) in base)
        print(f"  n_train={n:>3}: encoder {enc[n]:.3f}/{e:.3f} ({enc[n] / ORACLE_MEDIAN:.1f}x oracle) | baselines: {cmp}")

    n0 = sizes[0]
    pod_rep = {k: min(base[("pod36", n0)][k], base[("pod136", n0)][k]) for k in base[("pod36", n0)]}
    enc_rep = {x["repeat"]: x["median_field"] for x in rows if x["method"] == "encoder" and x["n_train"] == n0}
    pod_mean = min(np.mean(list(base[("pod36", n0)].values())), np.mean(list(base[("pod136", n0)].values())))
    a_mean = enc[n0] <= 0.5 * pod_mean
    a_reps = sum(enc_rep[k] <= 0.5 * pod_rep[k] for k in enc_rep)
    crit_a = a_mean and a_reps >= 2
    print(f"\nCriterion A (n_train={n0}): encoder {enc[n0]:.3f} against half of the smaller POD value {0.5 * pod_mean:.3f}; "
          f"at most half of the repeat's POD in {a_reps} of {len(enc_rep)} repeats -> {'MET' if crit_a else 'NOT MET'}")
    b_sizes = [n for n in sizes if n >= 50 and enc[n] <= np.mean(list(base[("pod136", n)].values()))]
    crit_b = bool(b_sizes)
    print(f"Criterion B (dimension 36 at most POD-136's error at some size >= 50): sizes {b_sizes or 'none'} -> {'MET' if crit_b else 'NOT MET'}")
    nonfinite = any(not np.isfinite(x.get("median_field", 0.0)) for x in rows)
    verdict = "INCONCLUSIVE (non-finite errors)" if nonfinite else ("STRONG PASS" if crit_a and crit_b else "PASS" if crit_a or crit_b else "FAIL")
    print(f"Verdict: {verdict}")
    first10 = [n for n in sizes if enc[n] < FIELD_TOL]
    print(f"Smallest size with median field error below 10%: {first10[0] if first10 else 'none'}")
    write(rows, a.out, "gate2")


if __name__ == "__main__":
    main()
