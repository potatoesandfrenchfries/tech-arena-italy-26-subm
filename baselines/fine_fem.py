from compat_galerkin import fem

from .base import Method


class FineFEM(Method):
    """Reference: the full fine-mesh solve. Sets the accuracy ceiling and the cost to beat."""

    name = "fine_fem"

    def predict(self, family, inst, cav):
        return fem.solve_fine(cav, inst.omega2, inst.f)
