"""Tests for the learned-basis code added after Gate 2: step schedules and the projection loss, the mesh-free and grid-base
encoders, multi-mesh training, the fast inference path, the enriched Galerkin method and the design-loop helper."""
import numpy as np
import pytest
import torch

from baselines import CoarseWhitney, EncoderWhitney, GridBaseEncoder, MeshFreeEncoder
from baselines.enriched_galerkin import EnrichedGalerkin
from compat_galerkin import fem, metrics, oracle
from compat_galerkin.problems import Dataset, Family, FamilyConfig

TINY = dict(n=3, width=8, batch=3, steps=4)


@pytest.fixture(scope="module")
def family():
    return Family(FamilyConfig(refine=3, grid=16))


@pytest.fixture(scope="module")
def train(family):
    return family.dataset(6, seed=0)


@pytest.fixture(scope="module")
def test(family):
    return family.dataset(3, seed=1)


@pytest.fixture(scope="module")
def encoder(family, train):
    enc = EncoderWhitney(**TINY, seed=0)
    enc.fit(family, train)
    return enc


def logits_for(enc, family, inst):
    return enc._logits(enc._prepare_inputs(family, [inst]))[0]


def test_default_schedule_is_one_galerkin_stage():
    enc = EncoderWhitney(n=3, steps=7)
    assert enc.schedule == [("galerkin", 7)]


def test_projection_error_never_exceeds_galerkin_error(family, test, encoder):
    # the Galerkin solution lies in the span of the basis, and the projection is the best approximation in that span
    with torch.no_grad():
        for inst, e_ref in zip(test.instances, test.e_ref):
            p = oracle.OracleProblem.from_instance(family.cavity(inst), inst, e_ref)
            V = encoder._basis(logits_for(encoder, family, inst))
            assert float(encoder._proj_error(p, V)) <= float(encoder._error(p, V)) + 1e-9
    errs = encoder.projection_errors(family, test)
    assert len(errs) == len(test) and all(0 <= e <= 1.0001 for e in errs)


def test_two_stage_schedule_trains_and_keeps_pec(family, train, test):
    enc = EncoderWhitney(**{**TINY, "steps": 6}, seed=0, schedule=[("projection", 3), ("galerkin", 3)])
    info = enc.fit(family, train)
    assert np.isfinite(info["train_error"]) and info["dim"] > 0
    inst = test.instances[0]
    e = enc.predict(family, inst, family.cavity(inst))
    assert np.all(np.isfinite(e)) and np.linalg.norm(e[family.base.boundary_edges]) == 0.0


def test_fast_inference_path_matches_predict(family, test, encoder):
    from experiments.study4b import Runner

    runner = Runner(encoder, family, encoder.p0.coords)
    for inst in test.instances:
        cav = family.cavity(inst)
        slow = encoder.predict(family, inst, cav)[cav.interior_edges]
        fast = runner.predict(inst, cav)
        assert np.linalg.norm(slow - fast) / np.linalg.norm(slow) < 1e-8


def test_mesh_free_encoder_has_no_per_node_parameters(family, train):
    fine = Family(FamilyConfig(refine=4, grid=16))
    enc = MeshFreeEncoder(**TINY, seed=0)
    info = enc.fit(family, train)
    assert info["params"] == sum(p.numel() for p in enc.net.parameters()) + 3 * 2 + 3  # centres and log-widths only
    coarse_xy = torch.as_tensor(family.base.mesh.p.T, dtype=torch.float64)
    fine_xy = torch.as_tensor(fine.base.mesh.p.T, dtype=torch.float64)
    assert enc.base_logits(fine_xy).shape == (fine.base.n_nodes, 3)
    assert torch.allclose(enc.base_logits(coarse_xy), enc.base_logits())  # the same function of the coordinates


def test_grid_base_encoder_samples_its_grid_exactly_at_grid_points(family, train):
    enc = GridBaseEncoder(**TINY, seed=0, grid=8)
    info = enc.fit(family, train)
    assert info["params"] == sum(p.numel() for p in enc.net.parameters()) + 3 * 8 * 8
    xs = np.linspace(0, 1, 8)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    xy = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
    out = enc.base_logits(xy).reshape(8, 8, 3).permute(2, 0, 1)  # (n, x index, y index)
    assert torch.allclose(out, enc.grid_base, atol=1e-10)
    fine = Family(FamilyConfig(refine=4, grid=16))
    assert enc.base_logits(torch.as_tensor(fine.base.mesh.p.T, dtype=torch.float64)).shape == (fine.base.n_nodes, 3)


def test_grid_base_multi_mesh_training_runs(family, train):
    fine = Family(FamilyConfig(refine=4, grid=16))
    ds_fine = fine.dataset(3, seed=0)
    enc = GridBaseEncoder(**TINY, seed=0, grid=8)
    info = enc.fit_multi([(family, Dataset(train.instances[:3], train.e_ref[:3])), (fine, ds_fine)])
    assert np.isfinite(info["train_error"]) and info["dim"] > 0
    inst = ds_fine.instances[0]
    from experiments.study5c import EncoderOnMesh

    e = EncoderOnMesh(enc, enc.p0.coords, family.base.mesh, "nearest").predict(fine, inst, fine.cavity(inst))
    assert np.all(np.isfinite(e)) and np.linalg.norm(e[fine.base.boundary_edges]) == 0.0


def test_enriched_galerkin_is_a_galerkin_solution(family, test):
    coarse = CoarseWhitney(2)
    coarse.fit(family, None)
    inst, e_ref = test.instances[0], test.e_ref[0]
    cav = family.cavity(inst)
    # a field that already contains the exact solution: the Galerkin solution in the enriched space is exact
    exact = EnrichedGalerkin(coarse, lambda fam, i, c: fem.solve_fine(c, i.omega2, i.f))
    r = metrics.evaluate(cav, inst, exact.predict(family, inst, cav), e_ref)
    assert r["rel_err"] < 1e-4 and r["power_balance"] < 1e-9 and r["pec_violation"] == 0
    # an arbitrary field: still a Galerkin solution, so power balance and PEC hold whatever the field is
    rng = np.random.default_rng(0)
    junk = EnrichedGalerkin(coarse, lambda fam, i, c: rng.standard_normal(c.n_edges) + 1j * rng.standard_normal(c.n_edges))
    r = metrics.evaluate(cav, inst, junk.predict(family, inst, cav), e_ref)
    assert r["power_balance"] < 1e-9 and r["pec_violation"] == 0


def test_random_search_regret_falls_with_more_calls():
    from experiments.study9 import random_regret

    fom = np.random.default_rng(0).uniform(1, 2, 200)
    rng = np.random.default_rng(1)
    r = [random_regret(fom, c, rng, resamples=60) for c in (1, 10, 100, 200)]
    assert r[0] > r[1] > r[2] > r[3] - 1e-12 and abs(r[3]) < 1e-12
