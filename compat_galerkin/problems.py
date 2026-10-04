"""A parametric family of cavity problems shared by every method in the comparison.

All instances live on one fixed fine mesh. They vary in the dielectric disc (position, radius,
contrast), the frequency and the current source. Reference solutions come from the fine FEM.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from skfem import Basis, ElementTriP1

from . import fem


@dataclass
class FamilyConfig:
    refine: int = 4
    grid: int = 32  # resolution of the regular grid used by grid-based neural baselines
    eps_in: tuple = (2.0, 8.0)
    radius: tuple = (0.10, 0.25)
    center: tuple = (0.3, 0.7)
    omega_rel: tuple = (0.2, 0.6)  # main family: Re omega^2 = r * lambda_1(instance), r uniform; see README
    omega2: tuple = (2.0, 12.0)  # absolute Re omega^2, used only when omega_rel is None (includes resonance)
    tan_delta: float = 0.05  # loss: omega^2 -> omega^2 (1 + i tan_delta)
    src_center: tuple = (0.2, 0.8)
    src_sigma: float = 0.08


@dataclass
class Instance:
    eps_elem: np.ndarray
    omega2: complex
    f: np.ndarray  # load vector on all edges
    params: dict = field(default_factory=dict)


@dataclass
class Dataset:
    instances: list
    e_ref: np.ndarray  # (n, n_edges) complex fine-mesh reference solutions

    def __len__(self):
        return len(self.instances)


class GridSampler:
    """Maps between edge cochains and a cell-centred regular grid on the unit square.

    Arrays are indexed [i, j] with i along x and j along y. Fields are (2, n, n) = (Ex, Ey).
    """

    def __init__(self, cav, n):
        self.n = n
        g = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(g, g, indexing="ij")
        self.X, self.Y = X, Y
        pts = np.vstack([X.ravel(), Y.ravel()])
        assert (cav.mesh.element_finder()(*pts) >= 0).all(), "grid points outside the mesh"
        self.probe = cav.basis.probes(pts).tocsr()  # (2 n^2, E), rows [Ex(all pts); Ey(all pts)]
        self.R = self._edge_map(cav)

    def _edge_map(self, cav):
        """Grid field -> edge coefficients: DOF_SIGN * E(midpoint) . (p_hi - p_lo), bilinear in the grid."""
        n = self.n
        lo, hi = cav.mesh.facets
        p = cav.mesh.p
        d = p[:, hi] - p[:, lo]
        s = np.clip((p[:, lo] + p[:, hi]) / 2 * n - 0.5, 0, n - 1 - 1e-9)
        i0, j0 = np.floor(s[0]).astype(int), np.floor(s[1]).astype(int)
        fx, fy = s[0] - i0, s[1] - j0
        rows, cols, vals = [], [], []
        e = np.arange(len(lo))
        for di, dj, w in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
            flat = (i0 + di) * n + (j0 + dj)
            for comp in range(2):
                rows.append(e)
                cols.append(comp * n * n + flat)
                vals.append(fem.DOF_SIGN * w * d[comp])
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(lo), 2 * n * n))

    def e_to_grid(self, e):
        return (self.probe @ e).reshape(2, self.n, self.n)

    def grid_to_e(self, field):
        return self.R @ np.asarray(field).reshape(-1)


class Family:
    def __init__(self, cfg=None):
        self.cfg = cfg or FamilyConfig()
        self.base = fem.build_cavity(fem.unit_square(self.cfg.refine))
        self.grid = GridSampler(self.base, self.cfg.grid)
        self._hats = {}

    def coarse_hats(self, level):
        """Interior P1 hat functions of the level-`level` coarse mesh, on the fine nodes (nodes x k)."""
        if level not in self._hats:
            coarse = fem.unit_square(level)
            P = Basis(coarse, ElementTriP1()).probes(self.base.mesh.p).tocsc()
            bnd = np.unique(coarse.facets[:, coarse.boundary_facets()])
            interior = np.setdiff1d(np.arange(coarse.p.shape[1]), bnd)
            self._hats[level] = P[:, interior].tocsr()
        return self._hats[level]

    def cavity(self, inst):
        return self.base.with_eps(inst.eps_elem)

    def sample(self, rng):
        c = self.cfg
        center = rng.uniform(*c.center, 2)
        eps = fem.inclusion_eps(self.base.mesh, rng.uniform(*c.eps_in), tuple(center), rng.uniform(*c.radius))
        src = rng.uniform(*c.src_center, 2)
        theta = rng.uniform(0, 2 * np.pi)
        direction = (np.cos(theta), np.sin(theta))
        f = fem.current_source(self.base, tuple(src), c.src_sigma, direction)
        params = dict(src=src, theta=theta, direction=direction)
        if c.omega_rel is not None:
            lam1 = fem.first_mode(self.base.with_eps(eps))
            ratio = rng.uniform(*c.omega_rel)
            params.update(lambda1=lam1, ratio=ratio)
            omega2 = ratio * lam1 * (1 + 1j * c.tan_delta)
        else:
            omega2 = rng.uniform(*c.omega2) * (1 + 1j * c.tan_delta)
        return Instance(eps, omega2, f, params)

    def dataset(self, n, seed=0):
        rng = np.random.default_rng(seed)
        instances = [self.sample(rng) for _ in range(n)]
        e_ref = np.stack([fem.solve_fine(self.cavity(i), i.omega2, i.f) for i in instances])
        return Dataset(instances, e_ref)

    def grid_inputs(self, inst):
        """Input channels for grid-based networks: eps, Re/Im omega^2, Jx, Jy, x, y. Shape (7, n, n)."""
        g, c = self.grid, self.cfg
        eps = inst.eps_elem[self.base.mesh.element_finder()(g.X.ravel(), g.Y.ravel())].reshape(g.X.shape)
        bump = np.exp(-((g.X - inst.params["src"][0]) ** 2 + (g.Y - inst.params["src"][1]) ** 2) / (2 * c.src_sigma**2))
        one = np.ones_like(g.X)
        chans = [eps, inst.omega2.real * one, inst.omega2.imag * one,
                 inst.params["direction"][0] * bump, inst.params["direction"][1] * bump, g.X, g.Y]
        return np.stack(chans).astype(np.float32)
