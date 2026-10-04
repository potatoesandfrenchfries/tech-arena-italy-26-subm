"""Post hoc check for Gate 1d: how well can exact gradients of coarse hats represent the curl-free
part of the solution?

The exact solution splits M-orthogonally into a curl-free part e_L = -(1/omega^2) grad(phi) and a
transverse part. phi solves a coercive Poisson-type problem with the epsilon-weighted mass matrix,
independent of omega:  (G^T M G) phi = G^T f  (interior nodes). This script computes e_L for the Gate 1
instance, its share of the solution at a few frequencies, and the relative M-norm error of its best
approximation by the gradients of the level-L coarse hat functions.

    uv run python -m experiments.diag_longitudinal
"""
import numpy as np
import scipy.sparse.linalg as spla

from compat_galerkin import fem
from compat_galerkin.problems import Family, FamilyConfig
from experiments.gate1 import build_instance

TAN = 0.05


def main():
    fam = Family(FamilyConfig(refine=4))
    cav, eps, f = build_instance(fam)
    idx = cav.interior_edges
    nodes = ~cav.boundary_node_mask
    Mn = cav.interior(cav.M).tocsc()
    Gall = cav.G.tocsr()[idx]
    Gn = Gall[:, np.flatnonzero(nodes)]
    fi = f[idx]
    phi = spla.spsolve((Gn.T @ Mn @ Gn).tocsc(), Gn.T @ fi)
    eL = Gn @ phi  # curl-free direction; the true curl-free part is -(1/omega^2) * eL
    mnorm = lambda x: float(np.sqrt(np.real(np.vdot(x, Mn @ x))))
    print(f"curl-free dimension (interior nodes): {int(nodes.sum())}")

    print("share of ||e_ref||^2 that is curl-free (exact):")
    for w2 in (6.5, 7.594, 9.0):
        e_ref = fem.solve_fine(cav, w2 * (1 + 1j * TAN), f)[idx]
        share = (mnorm(eL) / w2 / mnorm(e_ref)) ** 2
        print(f"  omega^2 = {w2:5.3f}: {share:.3f}")

    print("best approximation of the curl-free part by gradients of coarse hats (relative M-norm error):")
    for level in (1, 2, 3):
        H = (Gall @ fam.coarse_hats(level)).toarray()
        MH = Mn @ H
        c = np.linalg.lstsq(H.T @ MH, MH.T @ eL, rcond=None)[0]
        err = mnorm(eL - H @ c) / mnorm(eL)
        print(f"  level {level}: {H.shape[1]:3d} functions, error {err:.3f}")


if __name__ == "__main__":
    main()
