"""Study 9: a screen-and-certify design loop (level 0): fewer full-wave calls per finished design?

Designs are discs (centre, radius, permittivity) in the main-family cavity; the figure of merit is the field energy in a target
region. Surrogates screen a pool of random designs; the residual certificate gates which surrogate values are trusted.
Protocol and decision rules: README "Study 9 protocol". Usage:
    OMP_NUM_THREADS=8 uv run python -u -m experiments.study9 [--tasks 5] [--pools 10]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from baselines import FNO, EncoderWhitney, PODGalerkin
from compat_galerkin import certify, fem
from compat_galerkin.device import get_device
from compat_galerkin.problems import Dataset, Family, FamilyConfig, Instance
from experiments.study4a import collect
from experiments.study5a import FNOOnMesh
from experiments.study5c import EncoderOnMesh
from experiments.verification import BOX

LOW, HIGH = np.array([0.3, 0.3, 0.10, 2.0]), np.array([0.7, 0.7, 0.25, 8.0])  # cx, cy, radius, epsilon
OMEGA2 = (2.0, 3.2)
POOL, K, TAU, TOP_M, ALPHA, TARGET = 1000, 50, 0.2, 5, 0.1, 0.10
M_GRID = (1, 2, 5, 10, 20, 50)
C_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FULL_393K, SURROGATE_393K = 6.6, {"fno": 0.0265, "g12": 3.0}  # seconds, Studies 4b, 5b and 8


def write(rows, out, stem):
    Path(out).mkdir(exist_ok=True)
    path = Path(out) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = list(dict.fromkeys(k for x in rows for k in x))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def random_regret(true_fom, c, rng, resamples=200):
    """Expected regret of full-wave search on c random pool designs, estimated by resampling."""
    best = true_fom.max()
    c = int(min(max(round(c), 1), len(true_fom)))
    return float(np.mean([1 - true_fom[rng.choice(len(true_fom), c, replace=False)].max() / best for _ in range(resamples)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--pools", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=POOL)
    ap.add_argument("--ncal", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    device = get_device("auto")
    fam = Family(FamilyConfig(refine=4, grid=64))
    base, idx = fam.base, fam.base.interior_edges
    MR = base.interior(fem.region_mass(base, BOX))
    fom = lambda e_int: float(np.real(np.vdot(e_int, MR @ e_int)))
    tan = fam.cfg.tan_delta

    t0 = time.time()
    train = fam.dataset(400, seed=400)
    fno = FNO(epochs=a.epochs, batch=16, seed=0)
    fno.fit(fam, train, device=device)
    enc = EncoderWhitney(n=12, width=24, seed=0, schedule=[("galerkin", a.steps)])
    enc.fit(fam, train)
    pod = PODGalerkin(rank=136)
    pod.fit(fam, train)
    models = {"fno": FNOOnMesh(fno, device), "g12": EncoderOnMesh(enc, enc.p0.coords, base.mesh, "nearest"), "pod136": pod}
    print(f"trained the surrogates ({time.time() - t0:.0f}s)", flush=True)

    cal = fam.dataset(a.ncal, seed=600)
    rc, _, _, _ = collect(fam, cal, models, MR)
    qhat, qsel = {}, {}
    for s in models:
        sel = np.argsort(-rc[s]["fom"])[: max(1, a.ncal // 10)]
        qhat[s] = certify.conformal_quantile(rc[s]["eps"] / rc[s]["eta_D"], ALPHA)
        qsel[s] = certify.conformal_quantile(rc[s]["eps"][sel] / rc[s]["eta_D"][sel], ALPHA)
    print("calibration quantiles (random, selected): " + ", ".join(f"{s} {qhat[s]:.2f}/{qsel[s]:.2f}" for s in models), flush=True)

    rows = []
    for t in range(a.tasks):
        rt = np.random.default_rng(900 + t)
        src, theta = rt.uniform(0.2, 0.8, 2), rt.uniform(0, 2 * np.pi)
        direction = (np.cos(theta), np.sin(theta))
        omega2 = rt.uniform(*OMEGA2) * (1 + 1j * tan)
        f = fem.current_source(base, tuple(src), fam.cfg.src_sigma, direction)
        for k in range(a.pools):
            t1 = time.time()
            rng = np.random.default_rng(1000 + t * 10 + k)
            designs = rng.uniform(LOW, HIGH, (a.pool_size, 4))
            insts = [Instance(fem.inclusion_eps(base.mesh, d[3], (d[0], d[1]), d[2]), omega2, f,
                              dict(src=src, theta=theta, direction=direction)) for d in designs]
            true_fom = np.zeros(a.pool_size)
            sfom = {s: np.zeros(a.pool_size) for s in models}
            pred = {s: np.zeros((a.pool_size, len(idx)), dtype=complex) for s in models}
            for i, inst in enumerate(insts):
                cav = fam.cavity(inst)
                true_fom[i] = fom(fem.solve_fine(cav, omega2, f)[idx])
                for s, m in models.items():
                    pred[s][i] = m.predict(fam, inst, cav)[idx]
                    sfom[s][i] = fom(pred[s][i])
            best = true_fom.max()
            rr = np.random.default_rng(7 + t * 10 + k)
            c_reg = {c: random_regret(true_fom, c, rr) for c in C_GRID}
            for s in models:
                order = np.argsort(-sfom[s])
                for M in M_GRID:
                    rows.append({"task": t, "pool": k, "surrogate": s, "policy": f"A{M}", "calls": M,
                                 "regret": 1 - true_fom[order[:M]].max() / best})
                cand = order[:K]
                eta = np.array([certify.Residual(fam.cavity(insts[i]), insts[i]).eta_D(pred[s][i]) for i in cand])
                certified = qsel[s] * eta <= TAU
                est = np.where(certified, sfom[s][cand], true_fom[cand])
                chosen = cand[np.argsort(-est)[:TOP_M]]
                paid = set(cand[~certified].tolist()) | set(chosen.tolist())
                rows.append({"task": t, "pool": k, "surrogate": s, "policy": "B", "calls": len(paid),
                             "regret": 1 - true_fom[list(paid)].max() / best,
                             "certified_in_topK": int(certified.sum()),
                             "random_regret_same_calls": random_regret(true_fom, len(paid), rr)})
                rows.append({"task": t, "pool": k, "surrogate": s, "policy": "curse",
                             "bias_top1": float(sfom[s][order[0]] / true_fom[order[0]] - 1)})
            for c, v in c_reg.items():
                rows.append({"task": t, "pool": k, "surrogate": "none", "policy": f"C{c}", "calls": c, "regret": v})
            print(f"  task {t} pool {k}: best FoM {best:.3g}; "
                  + ", ".join(f"{s} A5 regret {next(r['regret'] for r in rows[::-1] if r.get('surrogate') == s and r.get('policy') == 'A5'):.3f}" for s in models)
                  + f" ({time.time() - t1:.0f}s)", flush=True)
        write(rows, a.out, "study9_partial")

    def mean(sel, key="regret"):
        v = [r[key] for r in rows if sel(r) and key in r]
        return float(np.mean(v)) if v else float("nan")

    c_mean = {c: mean(lambda r, c=c: r["policy"] == f"C{c}") for c in C_GRID}
    c_need = next((c for c in C_GRID if c_mean[c] <= TARGET), None)
    print("\nMean regret of full-wave random search by calls: " + ", ".join(f"{c}: {c_mean[c]:.3f}" for c in C_GRID))
    print(f"Calls for random search to reach regret <= {TARGET}: {c_need if c_need else '> 1000'}")
    for s in models:
        a_mean = {M: mean(lambda r, M=M, s=s: r["surrogate"] == s and r["policy"] == f"A{M}") for M in M_GRID}
        m_need = next((M for M in M_GRID if a_mean[M] <= TARGET), None)
        ratio = (m_need / c_need) if (m_need and c_need) else float("inf")
        works = "SCREENING WORKS" if ratio <= 0.5 else "PARTIAL" if ratio <= 1 else "NO"
        b_calls, b_reg = mean(lambda r, s=s: r["surrogate"] == s and r["policy"] == "B", "calls"), mean(lambda r, s=s: r["surrogate"] == s and r["policy"] == "B")
        b_cert = mean(lambda r, s=s: r["surrogate"] == s and r["policy"] == "B", "certified_in_topK")
        m_eq = next((M for M in M_GRID if M >= b_calls), M_GRID[-1])
        value = "CERTIFICATE ADDS VALUE" if b_reg <= 0.8 * a_mean[m_eq] else "NO GAIN"
        b_c = mean(lambda r, s=s: r["surrogate"] == s and r["policy"] == "B", "random_regret_same_calls")
        curse = mean(lambda r, s=s: r["surrogate"] == s and r["policy"] == "curse", "bias_top1")
        print(f"\n{s}: policy A mean regret by M: " + ", ".join(f"{M}: {a_mean[M]:.3f}" for M in M_GRID))
        print(f"  calls to reach regret <= {TARGET}: A needs {m_need if m_need else '> 50'} against {c_need if c_need else '> 1000'} for random search -> {works}")
        print(f"  policy B: mean calls {b_calls:.1f} (certified in top {K}: {b_cert:.1f}), mean regret {b_reg:.3f}; random search at the same calls {b_c:.3f}; "
              f"A at M = {m_eq}: {a_mean[m_eq]:.3f} -> {value}")
        print(f"  optimiser's curse (surrogate over full-wave figure of merit of the top candidate, minus 1): {curse:+.2f}")
        if s in SURROGATE_393K:
            full_cost = (c_need or 1000) * FULL_393K
            proj = POOL * SURROGATE_393K[s] + (m_need or 50) * FULL_393K
            print(f"  projection at 393k unknowns (not a measurement): screening {POOL} designs {POOL * SURROGATE_393K[s]:.0f} s plus "
                  f"{m_need or 50} calls = {proj:.0f} s, against {full_cost:.0f} s for random full-wave search")
    write(rows, a.out, "study9")


if __name__ == "__main__":
    main()
