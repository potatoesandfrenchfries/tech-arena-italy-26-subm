"""Residual-based error indicators and split conformal calibration for the verification study.

For a surrogate answer e_h the residual is r = f - (K - omega^2 M) e_h on the interior edges. It costs one
sparse matrix-vector product. Three relative indicators are computed:
  eta_D  ||r||_{D^-1} / ||f||_{D^-1}, D = diag(M): O(N), spectrally equivalent to the dual norm (primary)
  eta_M  the same with the exact M^-1 (reference)
  eta_2  the Euclidean relative residual
Split conformal calibration is applied to the ratio score eps / eta, giving the certified bound
B(x) = q_hat * eta(x) with marginal coverage P(eps <= B) >= 1 - alpha.
"""
import math

import numpy as np
import scipy.sparse.linalg as spla


class Residual:
    """Residual indicators for one instance. Factorises the interior mass matrix once (for eta_M)."""

    def __init__(self, cav, inst):
        idx = cav.interior_edges
        self.K = cav.interior(cav.K)
        self.M = cav.interior(cav.M).tocsc()
        self.f = inst.f[idx].astype(float)
        self.w2 = inst.omega2
        self.d = self.M.diagonal()
        self.lu = spla.splu(self.M)
        self.f_D = math.sqrt(np.sum(self.f**2 / self.d))
        self.f_M = math.sqrt(self.f @ self.lu.solve(self.f))
        self.f_2 = np.linalg.norm(self.f)

    def residual(self, e_int):
        return self.f - (self.K @ e_int - self.w2 * (self.M @ e_int))

    def eta_D(self, e_int):
        r = self.residual(e_int)
        return math.sqrt(np.sum(np.abs(r) ** 2 / self.d)) / self.f_D

    def indicators(self, e_int):
        r = self.residual(e_int)
        x = self.lu.solve(np.ascontiguousarray(r.real)) + 1j * self.lu.solve(np.ascontiguousarray(r.imag))
        return {"D": math.sqrt(np.sum(np.abs(r) ** 2 / self.d)) / self.f_D,
                "M": math.sqrt(max(np.real(np.vdot(r, x)), 0.0)) / self.f_M,
                "2": np.linalg.norm(r) / self.f_2}


def conformal_quantile(scores, alpha):
    """Split conformal quantile: the k-th smallest score, k = ceil((n + 1)(1 - alpha)).

    With fewer than k calibration scores the guarantee needs an infinite bound, so inf is returned."""
    s = np.sort(np.asarray(scores))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= len(s) else math.inf


def certificate(eps_cal, eta_cal, eps_test, eta_test, alpha, taus=()):
    """Calibrate on (eps_cal, eta_cal), certify the test instances.

    Returns q_hat, the empirical coverage of B = q_hat * eta on the test set, and for each accuracy target tau
    the acceptance rate (B <= tau, no full-wave call) and the false-accept rate (accepted with eps > tau)."""
    q = conformal_quantile(np.asarray(eps_cal) / np.asarray(eta_cal), alpha)
    B = q * np.asarray(eta_test)
    eps_test = np.asarray(eps_test)
    out = {"q_hat": q, "coverage": float(np.mean(eps_test <= B))}
    for tau in taus:
        acc = B <= tau
        out[f"accept_{tau}"] = float(np.mean(acc))
        out[f"false_accept_{tau}"] = float(np.mean(acc & (eps_test > tau)))
    return out
