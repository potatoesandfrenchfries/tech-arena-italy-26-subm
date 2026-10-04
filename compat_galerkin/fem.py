"""Fine-mesh operators for 2D time-harmonic Maxwell (in-plane E, TE) in a PEC cavity.

Unknowns are lowest-order Nedelec (edge) coefficients from scikit-fem. The equation is
(K - omega^2 M) e = f with K the curl-curl matrix and M the epsilon-weighted mass matrix.
"""
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import (
    Basis,
    BilinearForm,
    ElementTriN1,
    ElementTriP0,
    LinearForm,
    MeshTri,
)
from skfem.helpers import curl, dot

# scikit-fem's N1 coefficient on facet (a, b), a < b, is the circulation from b to a,
# i.e. u_a - u_b for a gradient. Checked in checks.dof_convention.
DOF_SIGN = -1.0


@dataclass
class Cavity:
    mesh: MeshTri
    basis: Basis
    K: sp.csr_matrix  # curl-curl, all edges
    M: sp.csr_matrix  # epsilon-weighted mass, all edges
    G: sp.csr_matrix  # edges x nodes, cochain of the gradient of a P1 function
    A: sp.csr_matrix  # edges x nodes, average of a P1 function over each edge
    eps_elem: np.ndarray
    boundary_edges: np.ndarray
    boundary_nodes: np.ndarray
    basis_p0: Basis

    def with_eps(self, eps_elem):
        """Same mesh and operators, new epsilon: only the mass matrix is reassembled."""
        return replace(self, M=_assemble_mass(self.basis, self.basis_p0, eps_elem), eps_elem=eps_elem)

    @property
    def n_edges(self):
        return self.K.shape[0]

    @property
    def n_nodes(self):
        return self.mesh.p.shape[1]

    @property
    def interior_edges(self):
        mask = np.ones(self.n_edges, dtype=bool)
        mask[self.boundary_edges] = False
        return np.flatnonzero(mask)

    @property
    def boundary_node_mask(self):
        mask = np.zeros(self.n_nodes, dtype=bool)
        mask[self.boundary_nodes] = True
        return mask

    def interior(self, mat):
        idx = self.interior_edges
        return mat[idx][:, idx]


def unit_square(refine=3):
    return MeshTri.init_symmetric().refined(refine)


def l_shape(refine=3):
    return MeshTri.init_lshaped().refined(refine)


def inclusion_eps(mesh, eps_in=4.0, center=(0.5, 0.5), radius=0.2):
    """Piecewise-constant epsilon: a dielectric disc in vacuum."""
    centroid = mesh.p[:, mesh.t].mean(axis=1)
    inside = np.hypot(centroid[0] - center[0], centroid[1] - center[1]) < radius
    return np.where(inside, eps_in, 1.0)


@BilinearForm
def _mass_form(u, v, w):
    return w["eps"] * dot(u, v)


def _assemble_mass(basis, basis_p0, eps_elem):
    return _mass_form.assemble(basis, eps=basis_p0.interpolate(eps_elem)).tocsr()


def build_cavity(mesh, eps_elem=None):
    if eps_elem is None:
        eps_elem = np.ones(mesh.t.shape[1])
    basis = Basis(mesh, ElementTriN1())
    basis_p0 = Basis(mesh, ElementTriP0(), quadrature=basis.quadrature)

    @BilinearForm
    def curl_curl(u, v, w):
        return curl(u) * curl(v)

    K = curl_curl.assemble(basis).tocsr()
    M = _assemble_mass(basis, basis_p0, eps_elem)

    lo, hi = mesh.facets
    ne, nn = mesh.facets.shape[1], mesh.p.shape[1]
    rows = np.r_[np.arange(ne), np.arange(ne)]
    cols = np.r_[lo, hi]
    # DOF_SIGN * (u_hi - u_lo)
    G = sp.csr_matrix((np.r_[-DOF_SIGN * np.ones(ne), DOF_SIGN * np.ones(ne)], (rows, cols)), shape=(ne, nn))
    A = sp.csr_matrix((np.full(2 * ne, 0.5), (rows, cols)), shape=(ne, nn))

    return Cavity(
        mesh=mesh,
        basis=basis,
        K=K,
        M=M,
        G=G,
        A=A,
        eps_elem=eps_elem,
        boundary_edges=mesh.boundary_facets(),
        boundary_nodes=np.unique(mesh.facets[:, mesh.boundary_facets()]),
        basis_p0=basis_p0,
    )


def current_source(cav, center=(0.3, 0.4), sigma=0.08, direction=(0.0, 1.0)):
    """Load vector of a Gaussian current bump, all edges."""
    d = np.asarray(direction, dtype=float)

    @LinearForm
    def load(v, w):
        x, y = w.x
        g = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2))
        return (d[0] * v[0] + d[1] * v[1]) * g

    return load.assemble(cav.basis)


def solve_fine(cav, omega2, f):
    """Full fine-mesh PEC solve. Returns the coefficient vector on all edges (boundary = 0)."""
    idx = cav.interior_edges
    A = (cav.interior(cav.K) - omega2 * cav.interior(cav.M)).tocsc()
    e = np.zeros(cav.n_edges, dtype=complex)
    e[idx] = spla.spsolve(A, f[idx].astype(complex))
    return e


def first_mode(cav, sigma=7.4):
    """Lowest positive eigenvalue omega^2 of K v = lambda M v on the interior edges.

    K has a large kernel (the gradients), whose eigenvalues all sit at zero. Shift-invert around
    sigma and ask for only the two eigenvalues closest to sigma: for this family the first two modes
    (lambda_1, lambda_2 in about [5, 10]) are closer to sigma than the kernel is, so the degenerate
    kernel cluster, which ARPACK cannot converge on, is never requested. The starting vector is
    fixed for reproducibility. Falls back to a dense solve if ARPACK does not converge.
    """
    K, M = cav.interior(cav.K).tocsc(), cav.interior(cav.M).tocsc()
    v0 = np.random.default_rng(0).standard_normal(K.shape[0])
    try:
        w = spla.eigsh(K, k=2, M=M, sigma=sigma, which="LM", v0=v0, return_eigenvectors=False)
    except spla.ArpackNoConvergence:
        import scipy.linalg as sl

        w = sl.eigh(K.toarray(), M.toarray(), eigvals_only=True)
    w = w[w > 1e-3]
    if len(w) == 0:
        raise RuntimeError("no positive eigenvalue found")
    return float(w.min())


def region_mass(cav, box):
    """Mass matrix (all edges) weighted by the indicator of box = (x0, x1, y0, y1), via element centroids.

    e^H M_R e is the field energy inside the box (with weight 1, not epsilon)."""
    x0, x1, y0, y1 = box
    cx, cy = cav.mesh.p[:, cav.mesh.t].mean(axis=1)
    ind = ((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)).astype(float)
    return _assemble_mass(cav.basis, cav.basis_p0, ind)
