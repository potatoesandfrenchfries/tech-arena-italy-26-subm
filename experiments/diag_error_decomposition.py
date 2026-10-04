"""Post hoc diagnostic for Gate 1c: where does the remaining Galerkin error come from?

Gate 1c fixed the reduced spectrum near the resonance but left a frequency-independent error floor
of about 0.29 (projection error 0.03). This retrains two Gate 1c spaces (same seeds and steps) and,
for each, reports:

  1. capture      - how well the two true resonant eigenvectors lie in the space (sin of the principal
                    angles between the true pair and the space);
  2. alignment    - sin of the principal angles between the true pair and the reduced eigenvectors whose
                    eigenvalues sit at the resonance;
  3. spectrum     - the lowest eight reduced physical eigenvalues (the loss controls only four);
  4. decomposition- the true solution and the Galerkin error expanded in the exact fine-mesh
                    eigenbasis, grouped into curl-free (kernel), the resonant pair, and the other
                    transverse modes, as fractions of ||e_ref||^2 (error fractions sum to rel_err^2).

    uv run python -m experiments.diag_error_decomposition [--steps 500]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

from compat_galerkin import fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig, Instance
from experiments.diag_band_spectrum import BAND, ZERO, reduced_eigs
from experiments.gate1 import build_instance, frequencies
from experiments.gate1b import HELD, STARTS, TAN, TRAIN, Band, evaluate
from experiments.gate1c import K_TARGETS, train

CONFIGS = [(10.0, 16), (0.1, 16)]  # (w, n): the best Gate 1c run, and a single-mode run for contrast
GROUPS = ["kernel", "pair", "other"]


def sin_angles(A, B, M):
    """sin of the principal angles between the M-orthonormal column spaces of A and B (A is the smaller)."""
    s = np.linalg.svd(A.T @ (M @ B), compute_uv=False)
    return np.sqrt(np.clip(1 - s**2, 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    _, modes = frequencies(cav)
    lam1 = modes[0]
    hw = TAN * lam1
    Kn, Mn = cav.interior(cav.K).tocsc(), cav.interior(cav.M).tocsc()
    w_all, X = sl.eigh(Kn.toarray(), Mn.toarray())  # X is M-orthonormal
    group_idx = {"kernel": w_all <= ZERO, "pair": abs(w_all - lam1) < 1e-3 * lam1}
    group_idx["other"] = ~(group_idx["kernel"] | group_idx["pair"])
    Phi = X[:, group_idx["pair"]]
    lam = torch.as_tensor(w_all[w_all > ZERO][:K_TARGETS], dtype=torch.float64)
    print(f"true pair at {lam1:.3f} ({Phi.shape[1]} vectors); kernel dim {int(group_idx['kernel'].sum())}", flush=True)

    inst0 = Instance(eps, TRAIN[0] * (1 + 1j * TAN), f, {})
    p = oracle.OracleProblem.from_instance(cav, inst0, fem.solve_fine(cav, inst0.omega2, f))
    tr, he = Band(p, cav, f, TRAIN), Band(p, cav, f, HELD)
    freqs = np.concatenate([TRAIN, HELD])
    R = np.concatenate([tr.R.numpy(), he.R.numpy()], axis=1)

    rows = []
    for w, n in CONFIGS:
        t0 = time.time()
        runs = [train(tr, p, lam, n, w, kind, seed, a.steps) for kind, seed in STARTS]
        b = min(runs, key=lambda r: r["final"])
        with torch.no_grad():
            Q = oracle.lifted_basis(p, b["logits"])
            ev = evaluate(tr, he, Q)
        Qn = Q.numpy()
        s, U = np.linalg.eigh(Qn.T @ (Mn @ Qn))
        keep = s > 1e-8 * s.max()
        Qo = Qn @ (U[:, keep] / np.sqrt(s[keep]))  # M-orthonormal basis of the space
        mu, Y = np.linalg.eigh(Qo.T @ (Kn @ Qo))
        phys = mu > ZERO
        near = phys & (abs(mu - lam1) < hw)
        capture = sin_angles(Phi, Qo, Mn)
        align = sin_angles(Phi, Qo @ Y[:, near], Mn) if near.sum() >= Phi.shape[1] else np.array([np.nan])

        print(f"\nw={w}, n={n}  (reproduces Gate 1c: held gal max {ev['held_gal_max']:.3f}, energy max "
              f"{ev['held_energy_max']:.3f}; {time.time() - t0:.0f}s)")
        print(f"  capture sin(angle) of true pair in space: {np.round(capture, 4).tolist()}")
        print(f"  alignment sin(angle) reduced resonant eigenvectors vs true pair "
              f"({int(near.sum())} reduced modes at the resonance): {np.round(align, 4).tolist()}")
        print(f"  lowest 8 reduced physical eigenvalues: {np.round(mu[phys][:8], 3).tolist()}")

        per = []
        for i, w2 in enumerate(freqs):
            e_gal = lift.reduced_solve(Q, p.K, p.M, p.f, complex(w2 * (1 + 1j * TAN))).numpy()
            a_ref, a_gal = X.T @ (Mn @ R[:, i]), X.T @ (Mn @ e_gal)
            den = np.sum(abs(a_ref) ** 2)
            row = {"w": w, "n": n, "omega2": float(w2), "heldout": bool(i >= len(TRAIN))}
            for g in GROUPS:
                row[f"ref_{g}"] = float(np.sum(abs(a_ref[group_idx[g]]) ** 2) / den)
                row[f"err_{g}"] = float(np.sum(abs(a_gal - a_ref)[group_idx[g]] ** 2) / den)
            row["rel_err"] = float(np.sqrt(sum(row[f"err_{g}"] for g in GROUPS)))
            per.append(row)
            rows.append(row)
        mean = lambda k: float(np.mean([r[k] for r in per]))
        print("  share of ||e_ref||^2 in each group (mean over 17 frequencies): "
              + ", ".join(f"{g} {mean('ref_' + g):.3f}" for g in GROUPS))
        print("  share of ||e_ref||^2 that is Galerkin error (mean over 17 frequencies): "
              + ", ".join(f"{g} {mean('err_' + g):.4f}" for g in GROUPS) + f" | rel_err {mean('rel_err'):.3f}")
        near_res = min(per, key=lambda r: abs(r["omega2"] - lam1))
        far = max(per, key=lambda r: abs(r["omega2"] - lam1))
        for tag, r in (("nearest resonance", near_res), ("farthest from resonance", far)):
            print(f"  at omega^2={r['omega2']:.3f} ({tag}): true solution " +
                  ", ".join(f"{g} {r['ref_' + g]:.3f}" for g in GROUPS) + " | error " +
                  ", ".join(f"{g} {r['err_' + g]:.4f}" for g in GROUPS) + f" | rel_err {r['rel_err']:.3f}", flush=True)

        # Intervention: add exact gradients of coarse hat functions (a larger reduced 0-form space).
        Gi = cav.G.tocsr()[cav.interior_edges]
        for level in (1, 2, 3):
            grads = (Gi @ fam.coarse_hats(level)).toarray()
            Qa = lift.compress(torch.cat([Q, torch.as_tensor(grads, dtype=lift.DTYPE)], dim=1), p.M)
            with torch.no_grad():
                ea = evaluate(tr, he, Qa)
            print(f"  + gradients of {grads.shape[1]:3d} level-{level} hats: dim {Qa.shape[1]:4d} | held gal mean "
                  f"{ea['held_gal_mean']:.3f} max {ea['held_gal_max']:.3f} | proj max {ea['held_proj_max']:.3f} "
                  f"| energy max {ea['held_energy_max']:.3f}", flush=True)
            eg = reduced_eigs(p, Qa)
            eg = eg[eg > ZERO]
            print(f"      reduced physical eigenvalues: in band {np.round(eg[(eg >= BAND[0]) & (eg <= BAND[1])], 3).tolist()}, lowest 6 {np.round(eg[:6], 3).tolist()}", flush=True)
            rows.append({"w": w, "n": n, "omega2": np.nan, "heldout": True, "augment_level": level,
                         "augment_funcs": grads.shape[1], "aug_dim": Qa.shape[1],
                         "held_gal_mean": ea["held_gal_mean"], "held_gal_max": ea["held_gal_max"],
                         "held_proj_max": ea["held_proj_max"], "held_energy_max": ea["held_energy_max"]})

    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"diag_decomp_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
