import numpy as np
import pytest
import scipy.linalg as sl

from compat_galerkin import fem
from compat_galerkin.problems import Family, FamilyConfig


@pytest.fixture(scope="module")
def family():
    return Family(FamilyConfig(refine=3, grid=16))


def dense_first_mode(cav):
    w = sl.eigh(cav.interior(cav.K).toarray(), cav.interior(cav.M).toarray(), eigvals_only=True)
    return w[w > 1e-6].min()


@pytest.mark.parametrize("eps_in,radius,center", [
    (2.0, 0.10, (0.3, 0.3)),   # weakest perturbation: first mode near the empty-cavity value
    (8.0, 0.25, (0.5, 0.5)),   # strongest perturbation: lowest first mode in the family
    (5.0, 0.18, (0.7, 0.4)),
])
def test_first_mode_matches_dense(family, eps_in, radius, center):
    cav = family.base.with_eps(fem.inclusion_eps(family.base.mesh, eps_in, center, radius))
    assert fem.first_mode(cav) == pytest.approx(dense_first_mode(cav), rel=1e-6)


def test_default_band_is_relative_and_sub_resonant(family):
    ds = family.dataset(12, seed=1)
    lo, hi = family.cfg.omega_rel
    for inst in ds.instances:
        lam1, ratio = inst.params["lambda1"], inst.params["ratio"]
        assert lo <= ratio <= hi
        assert inst.omega2.real == pytest.approx(ratio * lam1)
        assert inst.omega2.real < lam1
        assert inst.omega2.imag == pytest.approx(family.cfg.tan_delta * inst.omega2.real)


def test_first_mode_of_sampled_instances_is_correct(family):
    ds = family.dataset(4, seed=2)
    for inst in ds.instances:
        assert inst.params["lambda1"] == pytest.approx(dense_first_mode(family.cavity(inst)), rel=1e-6)


def test_absolute_band_still_available():
    fam = Family(FamilyConfig(refine=3, grid=16, omega_rel=None, omega2=(2.0, 12.0)))
    rng = np.random.default_rng(0)
    for _ in range(5):
        inst = fam.sample(rng)
        assert 2.0 <= inst.omega2.real <= 12.0
        assert "ratio" not in inst.params
