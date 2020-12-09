from sbi.inference.base import NeuralInference  # noqa: F401
from sbi.inference.snpe.snpe_c import SNPE_C  # noqa: F401

SNPE = APT = SNPE_C

__all__ = ["SNPE_C", "SNPE", "APT"]
