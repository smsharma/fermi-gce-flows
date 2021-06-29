from typing import Optional

import torch
from torch import Tensor, nn

from sbi import utils as utils
from sbi.types import Shape

class DirectPosterior:

    def __init__(
        self,
        neural_net: nn.Module,
        prior,
    ):
        self.net = neural_net
        self.prior = prior

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:

        num_samples = torch.Size(sample_shape).numel()

        self.net.eval()

        # Rejection sampling.
        samples, _ = utils.sample_posterior_within_prior(
            self.net,
            self.prior,
            x,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
        )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))