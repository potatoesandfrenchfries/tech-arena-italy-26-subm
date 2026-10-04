"""Metrics shared by all methods. Every quantity is computed from the raw predicted edge vector.

gauss_fine tests the Gauss law against ALL fine-mesh interior nodal functions, so it measures the
true divergence error of the prediction. A reduced space only satisfies the weak Gauss law for
test functions inside its own 0-form space (see checks.gate0), so reduced methods are not expected
to reach machine precision there. gauss_r{L} restricts the test functions to the hat functions of
the level-L coarse mesh (nested in the fine mesh): a compatible space built from level-L0 hats is
exact for every L <= L0, a non-compatible one (POD, FNO) is not.
"""
import numpy as np


def _norm(x):
    return np.linalg.norm(x)


def evaluate(cav, inst, e, e_ref, hats=None):
    K, M, G = cav.K, cav.M, cav.G
    w2, f = inst.omega2, inst.f
    nodes = ~cav.boundary_node_mask
    idx = cav.interior_edges
    Me = M @ e

    def mnorm(x):
        return np.sqrt(np.real(np.vdot(x, M @ x)))

    gt_f = (G.T @ f)[nodes]
    gt_me = (G.T @ Me)[nodes]
    power_lhs = np.vdot(e, K @ e - w2 * Me)
    power_rhs = np.vdot(e, f)
    energy, energy_ref = np.real(np.vdot(e, Me)), np.real(np.vdot(e_ref, M @ e_ref))
    out = {
        "rel_err": mnorm(e - e_ref) / mnorm(e_ref),
        "residual": _norm((K @ e - w2 * Me - f)[idx]) / _norm(f[idx]),
        "gauss_fine": _norm(w2 * gt_me + gt_f) / (_norm(gt_f) + abs(w2) * _norm(gt_me)),
        "power_balance": abs(power_lhs - power_rhs) / abs(power_rhs),
        "pec_violation": _norm(e[cav.boundary_edges]) / _norm(e),
        "energy_err": abs(energy - energy_ref) / energy_ref,
    }
    resid = G.T @ (w2 * Me + f)
    for level, P in (hats or {}).items():
        ref = _norm(P.T @ (G.T @ f)) + abs(w2) * _norm(P.T @ (G.T @ Me))
        out[f"gauss_r{level}"] = _norm(P.T @ resid) / ref
    return out
