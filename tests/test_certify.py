import numpy as np
import pytest

from compat_galerkin import certify, fem
from compat_galerkin.problems import Family, FamilyConfig


def test_conformal_quantile_index():
    s = np.arange(1.0, 10.0)  # n = 9
    assert certify.conformal_quantile(s, 0.1) == 9.0  # ceil(10 * 0.9) / 9 = 1: the maximum
    assert certify.conformal_quantile(np.arange(1.0, 101.0), 0.1) == 91.0  # ceil(101 * 0.9) = 91


def test_marginal_coverage_and_false_accept_on_synthetic_data():
    rng = np.random.default_rng(0)
    covs, fas = [], []
    for _ in range(200):
        eta_c, eta_t = rng.lognormal(-3, 1, 300), rng.lognormal(-3, 1, 300)
        eps_c = eta_c * rng.lognormal(0, 0.5, 300)
        eps_t = eta_t * rng.lognormal(0, 0.5, 300)
        out = certify.certificate(eps_c, eta_c, eps_t, eta_t, 0.1, taus=(0.05,))
        covs.append(out["coverage"])
        fas.append(out["false_accept_0.05"])
    assert 0.90 <= np.mean(covs) <= 0.92  # exchangeable data: valid and not very conservative
    assert np.mean(fas) <= 0.1  # a false accept implies eps > B, so its rate is at most alpha


@pytest.fixture(scope="module")
def family():
    return Family(FamilyConfig(refine=3, grid=16))


def test_exact_solution_has_zero_residual_and_indicators_are_equivalent(family):
    inst = family.dataset(1, seed=3).instances[0]
    cav = family.cavity(inst)
    e = fem.solve_fine(cav, inst.omega2, inst.f)[cav.interior_edges]
    res = certify.Residual(cav, inst)
    ind = res.indicators(e)
    assert max(ind.values()) < 1e-10
    rng = np.random.default_rng(0)
    pert = e + 0.1 * np.linalg.norm(e) * rng.standard_normal(len(e)) / np.sqrt(len(e))
    ind = res.indicators(pert)
    assert ind["D"] == pytest.approx(res.indicators(pert)["D"]) and res.eta_D(pert) == pytest.approx(ind["D"])
    assert 0.2 < ind["D"] / ind["M"] < 5 and 0.2 < ind["2"] / ind["D"] < 50  # equivalent up to mesh-dependent constants


def test_region_mass(family):
    base = family.base
    whole = fem.region_mass(base, (0, 1, 0, 1))
    assert abs(whole - base.M).max() < 1e-12  # eps = 1 on the base cavity
    part = fem.region_mass(base, (0.65, 0.85, 0.15, 0.35))
    x = np.random.default_rng(0).standard_normal(base.n_edges)
    assert 0 < x @ (part @ x) < x @ (whole @ x)
