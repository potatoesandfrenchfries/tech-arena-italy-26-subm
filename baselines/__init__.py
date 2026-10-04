from .coarse_whitney import CoarseWhitney
from .encoder_whitney import EncoderWhitney, MeshFreeEncoder
from .grid_base_encoder import GridBaseEncoder
from .fine_fem import FineFEM
from .fno import FNO
from .pod_galerkin import PODGalerkin
from .shared_whitney import SharedWhitney

REGISTRY = {
    "fine_fem": FineFEM,
    "coarse_r1": lambda: CoarseWhitney(1),
    "coarse_r2": lambda: CoarseWhitney(2),
    "coarse_r3": lambda: CoarseWhitney(3),
    "pod_16": lambda: PODGalerkin(16),
    "pod_64": lambda: PODGalerkin(64),
    "fno": FNO,
}
