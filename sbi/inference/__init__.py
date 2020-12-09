from sbi.inference.base import (  # noqa: F401
    NeuralInference,
    check_if_proposal_has_default_x,
    infer,
    simulate_for_sbi,
)

from sbi.inference.snpe.snpe_c import SNPE_C  # noqa: F401

SNPE = APT = SNPE_C

__all__ = ["SNPE_C", "SNPE", "APT"]
