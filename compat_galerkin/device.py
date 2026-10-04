import torch


def get_device(prefer="auto"):
    """'auto' picks CUDA when available, else CPU. MPS is not used: it has no float64/complex128,
    which the reduced Galerkin solves need. Neural baselines run in float32 on this device."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)
