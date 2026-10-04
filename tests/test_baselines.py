import numpy as np
import pytest

from baselines import CoarseWhitney, FineFEM, FNO, PODGalerkin
from compat_galerkin import fem, metrics
from compat_galerkin.problems import Family, FamilyConfig


@pytest.fixture(scope="module")
def family():
    return Family(FamilyConfig(refine=3, grid=16))


@pytest.fixture(scope="module")
def data(family):
    return family.dataset(10, seed=0), family.dataset(4, seed=1)


def score(method, family, test):
    out = []
    for inst, e_ref in zip(test.instances, test.e_ref):
        cav = family.cavity(inst)
        out.append(metrics.evaluate(cav, inst, method.predict(family, inst, cav), e_ref))
    return out


def test_with_eps_matches_direct_build(family):
    eps = fem.inclusion_eps(family.base.mesh, 5.0, (0.4, 0.6), 0.15)
    direct = fem.build_cavity(family.base.mesh, eps)
    assert abs(family.base.with_eps(eps).M - direct.M).max() < 1e-14


def test_grid_probe_layout_and_edge_map(family):
    g, cav = family.grid, family.base
    u = cav.mesh.p[0]  # grad u = (1, 0)
    field = g.e_to_grid(cav.G @ u)
    assert np.allclose(field[0], 1) and np.allclose(field[1], 0)
    assert np.allclose(g.grid_to_e(field), cav.G @ u)  # exact for a constant field


def test_fine_fem_is_exact(family, data):
    res = score(FineFEM(), family, data[1])
    assert max(r["rel_err"] for r in res) < 1e-12
    assert max(r["residual"] for r in res) < 1e-10
    assert max(r["gauss_fine"] for r in res) < 1e-8


def test_coarse_whitney(family, data):
    m = CoarseWhitney(coarse_refine=2)
    info = m.fit(family, data[0])
    assert info["dim"] > 0
    res = score(m, family, data[1])
    assert max(r["pec_violation"] for r in res) == 0
    assert np.median([r["rel_err"] for r in res]) < 1.0  # better than predicting zero


def test_coarse_whitney_improves_with_dimension(family, data):
    errs = []
    for r in (1, 2):
        m = CoarseWhitney(r)
        m.fit(family, data[0])
        errs.append(np.median([x["rel_err"] for x in score(m, family, data[1])]))
    assert errs[1] < errs[0]


def test_pod_reproduces_training_snapshots(family, data):
    train, _ = data
    m = PODGalerkin(rank=1000)  # keeps every direction, so training solutions lie in the span
    m.fit(family, train)
    res = score(m, family, train)
    assert max(r["rel_err"] for r in res) < 1e-6


def test_fno_runs_end_to_end(family, data):
    m = FNO(epochs=2, batch=5, width=8, modes=4, layers=2)
    info = m.fit(family, data[0])
    assert np.isfinite(info["final_train_loss"]) and info["roundtrip_floor"] < 0.5
    res = score(m, family, data[1])
    assert all(np.isfinite(r["rel_err"]) for r in res)


def test_gauss_exact_at_and_below_own_level_only(family, data):
    # compatible space from level-2 hats: exact for level-1 and level-2 test functions, not level 3
    hats = {l: family.coarse_hats(l) for l in (1, 2, 3)}
    m = CoarseWhitney(2)
    m.fit(family, data[0])
    inst, e_ref = data[1].instances[0], data[1].e_ref[0]
    r = metrics.evaluate(family.cavity(inst), inst, m.predict(family, inst, family.cavity(inst)), e_ref, hats)
    assert r["gauss_r1"] < 1e-9 and r["gauss_r2"] < 1e-9
    assert r["gauss_r3"] > 1e-3
    # POD has no such structure
    pod = PODGalerkin(8)
    pod.fit(family, data[0])
    rp = metrics.evaluate(family.cavity(inst), inst, pod.predict(family, inst, family.cavity(inst)), e_ref, hats)
    assert rp["gauss_r1"] > 1e-3


def test_shared_whitney_fits_and_predicts(family):
    from baselines import SharedWhitney

    train = family.dataset(6, seed=11)
    m = SharedWhitney(n=4, steps=15, batch=3, starts=(("smooth", 0),))
    info = m.fit(family, train)
    assert info["dim"] == 10 and np.isfinite(info["train_error"])
    inst = train.instances[0]
    cav = family.cavity(inst)
    e = m.predict(family, inst, cav)
    assert e.shape == (cav.n_edges,) and np.all(np.isfinite(e))
    assert np.linalg.norm(e[cav.boundary_edges]) == 0.0  # PEC: boundary coefficients stay zero
