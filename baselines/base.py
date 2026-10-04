"""Common interface so every method is trained, timed and scored identically."""


class Method:
    name = "method"

    def fit(self, family, train, device=None, **kwargs):
        """Train on a Dataset. Return a dict of facts about the fit (dimension, floor, ...)."""
        return {}

    def predict(self, family, inst, cav):
        """Return complex edge coefficients (all edges) for one instance. cav = family.cavity(inst)."""
        raise NotImplementedError
