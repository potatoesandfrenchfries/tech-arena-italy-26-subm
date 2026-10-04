import numpy as np
import torch

from compat_galerkin import checks, fem, lift, oracle
from compat_galerkin.problems import Family, FamilyConfig


def small_problem():
    fam = Family(FamilyConfig(refine=2, grid=8))
    inst = fam.sample(np.random.default_rng(0))
    inst.omega2 = 4.0 * (1 + 0.05j)
    cav = fam.cavity(inst)
    return oracle.OracleProblem.from_instance(cav, inst, fem.solve_fine(cav, inst.omega2, inst.f)), cav


def test_cholesky_compress_gives_same_solution_as_eigh():
    cav = fem.build_cavity(fem.unit_square(3))
    W = checks.random_pou(cav, n=5, seed=0)
    A, G = lift.to_torch_sparse(cav.A), lift.to_torch_sparse(cav.G)
    idx = torch.as_tensor(cav.interior_edges)
    V = lift.whitney_lift(W, A, G)[idx]
    K, M = (lift.to_torch_sparse(cav.interior(x)) for x in (cav.K, cav.M))
    f = torch.as_tensor(fem.current_source(cav)[cav.interior_edges])
    e1 = lift.reduced_solve(lift.compress(V, M), K, M, f, 4.0 + 0.2j)
    e2 = lift.reduced_solve(lift.compress_chol(V, M), K, M, f, 4.0 + 0.2j)
    assert torch.linalg.norm(e1 - e2) / torch.linalg.norm(e1) < 1e-6


def test_autograd_matches_finite_differences():
    p, _ = small_problem()
    logits = oracle.init_logits(p, 4, "smooth", 0).requires_grad_(True)
    galerkin = oracle.galerkin_error(p, logits)
    galerkin.backward()
    for k in [(10, 0), (20, 2), (7, 3)]:
        if p.boundary_mask[k[0]]:
            continue
        h = 1e-6
        up, dn = logits.detach().clone(), logits.detach().clone()
        up[k] += h
        dn[k] -= h
        fd = (oracle.galerkin_error(p, up) - oracle.galerkin_error(p, dn)) / (2 * h)
        assert abs(float(fd) - float(logits.grad[k])) < 1e-4 * max(1.0, abs(float(fd)))


def test_optimiser_reduces_error_and_error_is_at_least_projection_floor():
    p, _ = small_problem()
    out = oracle.optimise(p, n=4, kind="rbf", seed=0, steps=60)
    assert out["history"][-1] < out["history"][0]
    # Galerkin error can never beat the best approximation in the same space
    assert float(oracle.projection_error(p, out["logits"])) <= out["final"] + 1e-9
