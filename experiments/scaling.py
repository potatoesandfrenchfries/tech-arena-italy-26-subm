"""Scaling study: when is a reduced solve cheaper than a full-wave solve?

Protocol and decision rules are in the README ("Scaling study protocol"). Run single-threaded and with
no other jobs:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.scaling
"""
import argparse
import csv
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import torch

from baselines import CoarseWhitney
from compat_galerkin import fem, lift

RATIO, TAN, REPEATS, MEM_LIMIT = 0.4, 0.05, 7, 2e9
DENSE_Q = (16, 64, 128)
COARSE = (2, 3)


def median_time(fn, repeats=REPEATS):
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def breakeven_sweep(t_full, t_setup, t_small):
    """Smallest F with F * t_full > t_setup + F * t_small, or inf."""
    return math.inf if t_full <= t_small else math.floor(t_setup / (t_full - t_small)) + 1


class Reduced:
    """Per-geometry setup and per-frequency solve of the reduced system for a fixed dense basis Q."""

    def __init__(self, Q, K_int, f_int):
        self.Q = Q
        self.K_t = lift.to_torch_sparse(K_int)  # K depends on the mesh only; converted once, untimed
        self.f = torch.as_tensor(f_int, dtype=lift.DTYPE)
        self.Kr_fixed = Q.T @ torch.sparse.mm(self.K_t, Q)

    def setup(self, M_int, project_K=True):
        M_t = lift.to_torch_sparse(M_int)
        Mr = self.Q.T @ torch.sparse.mm(M_t, self.Q)
        Kr = self.Q.T @ torch.sparse.mm(self.K_t, self.Q) if project_K else self.Kr_fixed
        return Kr, Mr, self.Q.T @ self.f

    def solve(self, Kr, Mr, rhs, w2):
        A = (Kr - w2 * Mr).to(torch.complex128)
        y = torch.linalg.solve(A, rhs.to(torch.complex128))
        return torch.complex(self.Q @ y.real, self.Q @ y.imag)


def eta_D(K_int, M_int, d, f, f_D, w2, e):
    r = f - (K_int @ e - w2 * (M_int @ e))
    return math.sqrt(float(np.sum(np.abs(r) ** 2 / d))) / f_D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refines", default="4,5,6,7,8")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(1)
    info = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cpu": platform.processor()}
    print("system:", info, flush=True)

    rows = []
    for refine in [int(x) for x in a.refines.split(",")]:
        t_start = time.time()
        mesh = fem.unit_square(refine)
        base = fem.build_cavity(mesh)
        eps = fem.inclusion_eps(mesh, 4.0, (0.5, 0.5), 0.2)
        t_assemble = median_time(lambda: base.with_eps(eps), repeats=3)
        cav = base.with_eps(eps)
        f = fem.current_source(cav)
        lam1 = fem.first_mode(cav)
        w2 = RATIO * lam1 * (1 + 1j * TAN)
        idx = cav.interior_edges
        N = len(idx)
        K_int, M_int, f_int = cav.interior(cav.K), cav.interior(cav.M), f[idx]
        d = M_int.diagonal()
        f_D = math.sqrt(float(np.sum(f_int**2 / d)))
        t_full = median_time(lambda: fem.solve_fine(cav, w2, f))
        e_full = fem.solve_fine(cav, w2, f)[idx]
        t_eta = median_time(lambda: eta_D(K_int, M_int, d, f_int, f_D, w2, e_full))
        print(f"\nrefine {refine}: N = {N}, lambda_1 = {lam1:.3f}, assemble {1e3 * t_assemble:.1f} ms, "
              f"full solve {1e3 * t_full:.1f} ms, eta_D {1e3 * t_eta:.2f} ms", flush=True)
        rows.append({"refine": refine, "N": N, "method": "full", "q": 0, "t_assemble_ms": 1e3 * t_assemble,
                     "t_full_ms": 1e3 * t_full, "t_eta_ms": 1e3 * t_eta})

        bases = {}
        for q in DENSE_Q:
            if N * q * 8 > MEM_LIMIT:
                print(f"  dense_q{q}: skipped (basis would exceed 2 GB)")
                continue
            g = torch.Generator().manual_seed(0)
            bases[f"dense_q{q}"] = torch.linalg.qr(torch.randn(N, q, generator=g, dtype=lift.DTYPE))[0]
        for lvl in COARSE:
            m = CoarseWhitney(lvl)
            q = m.fit(SimpleNamespace(base=base), None)["dim"]
            if N * q * 8 > MEM_LIMIT:
                print(f"  coarse_r{lvl}: skipped (basis would exceed 2 GB)")
                continue
            bases[f"coarse_r{lvl}"] = m.Q

        for name, Q in bases.items():
            R = Reduced(Q, K_int, f_int)
            t_setup = median_time(lambda: R.setup(M_int, True))
            t_setup_fixed = median_time(lambda: R.setup(M_int, False))
            Kr, Mr, rhs = R.setup(M_int, True)
            t_small = median_time(lambda: R.solve(Kr, Mr, rhs, w2))
            single = t_setup + t_small + t_eta
            row = {"refine": refine, "N": N, "method": name, "q": Q.shape[1], "t_assemble_ms": 1e3 * t_assemble,
                   "t_full_ms": 1e3 * t_full, "t_eta_ms": 1e3 * t_eta, "t_setup_ms": 1e3 * t_setup,
                   "t_setup_fixedK_ms": 1e3 * t_setup_fixed, "t_small_ms": 1e3 * t_small,
                   "single_total_ms": 1e3 * single, "wins": bool(single < t_full),
                   "wins_fixedK": bool(t_setup_fixed + t_small + t_eta < t_full),
                   "sweep_Fstar": breakeven_sweep(t_full, t_setup, t_small),
                   "sweep_Fstar_fixedK": breakeven_sweep(t_full, t_setup_fixed, t_small),
                   "p_star": 1 - single / t_full}
            rows.append(row)
            print(f"  {name:<10} q={row['q']:4d} setup {row['t_setup_ms']:8.1f} ms (K fixed {row['t_setup_fixedK_ms']:8.1f}) "
                  f"small {row['t_small_ms']:7.2f} ms | single {row['single_total_ms']:8.1f} ms vs full "
                  f"{row['t_full_ms']:8.1f} -> {'WINS' if row['wins'] else 'loses'} | sweep F* {row['sweep_Fstar']} "
                  f"(K fixed {row['sweep_Fstar_fixedK']}) | p* {row['p_star']:.2f}", flush=True)
            del R, Kr, Mr, rhs
        del bases
        print(f"  (refine {refine} took {time.time() - t_start:.0f}s)", flush=True)

    print("\nLog-log slope of cost against N (all meshes run):")
    for key in ("t_assemble_ms", "t_full_ms", "t_eta_ms"):
        pts = [(r["N"], r[key]) for r in rows if r["method"] == "full"]
        print(f"  {key:<14} {np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]:.2f}")
    for name in sorted({r["method"] for r in rows} - {"full"}):
        pts = [(r["N"], r["t_setup_ms"], r["t_small_ms"]) for r in rows if r["method"] == name]
        if len(pts) >= 2:
            x = np.log([p[0] for p in pts])
            print(f"  {name:<10} setup {np.polyfit(x, np.log([p[1] for p in pts]), 1)[0]:.2f}  "
                  f"small(+reconstruct) {np.polyfit(x, np.log([p[2] for p in pts]), 1)[0]:.2f}")

    print("\nSmallest tested mesh where the single query wins (setup + small + eta_D < full):")
    for name in sorted({r["method"] for r in rows} - {"full"}):
        win = [r["N"] for r in rows if r["method"] == name and r["wins"]]
        winf = [r["N"] for r in rows if r["method"] == name and r["wins_fixedK"]]
        print(f"  {name:<10} per-geometry K projection: {min(win) if win else 'none'} | K projected once: {min(winf) if winf else 'none'}")

    Path(a.out).mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(Path(a.out) / f"scaling_{stamp}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote results/scaling_{stamp}.csv; system: {info}")


if __name__ == "__main__":
    main()
