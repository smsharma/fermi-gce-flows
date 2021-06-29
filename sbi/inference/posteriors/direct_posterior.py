# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, log, nn

from sbi import utils as utils
# from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import ScalarFloat, Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import (
    batched_first_of_batch,
    ensure_theta_batched,
    ensure_x_batched,
)


class DirectPosterior:
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNPE.<br/><br/>
    SNPE trains a neural network to directly approximate the posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.<br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/>
    - reject samples that lie outside of the prior bounds.<br/>
    - alternatively, if leakage is very high (which can happen for multi-round SNPE),
      sample from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self,
        neural_net: nn.Module,
        prior,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            rejection_sampling_parameters: Dictonary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
            sample_with_mcmc: Whether to sample with MCMC. Will always be `True` for SRE
                and SNL, but can also be set to `True` for SNPE if MCMC is preferred to
                deal with leakage over rejection sampling.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
        """

        self.net = neural_net
        self._prior = prior

    # def log_prob(
    #     self,
    #     theta: Tensor,
    #     x: Optional[Tensor] = None,
    #     x_aux: Optional[Tensor] = None,
    #     norm_posterior: bool = True,
    #     track_gradients: bool = False,
    # ) -> Tensor:
    #     r"""
    #     Returns the log-probability of the posterior $p(\theta|x).$

    #     Args:
    #         theta: Parameters $\theta$.
    #         x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
    #             back onto an `x_o` if previously provided for multi-round training, or
    #             to another default if set later for convenience, see `.set_default_x()`.
    #         norm_posterior: Whether to enforce a normalized posterior density.
    #             Renormalization of the posterior is useful when some
    #             probability falls out or leaks out of the prescribed prior support.
    #             The normalizing factor is calculated via rejection sampling, so if you
    #             need speedier but unnormalized log posterior estimates set here
    #             `norm_posterior=False`. The returned log posterior is set to
    #             -∞ outside of the prior support regardless of this setting.
    #         track_gradients: Whether the returned tensor supports tracking gradients.
    #             This can be helpful for e.g. sensitivity analysis, but increases memory
    #             consumption.

    #     Returns:
    #         `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
    #         support of the prior, -∞ (corresponding to 0 probability) outside.

    #     """

    #     # TODO Train exited here, entered after sampling?
    #     self.net.eval()

    #     with torch.set_grad_enabled(track_gradients):

    #         unnorm_log_prob = self.net.log_prob(theta, x, x_aux)

    #         # Force probability to be zero outside prior support.
    #         is_prior_finite = torch.isfinite(self._prior.log_prob(theta))

    #         masked_log_prob = torch.where(
    #             is_prior_finite,
    #             unnorm_log_prob,
    #             torch.tensor(float("-inf"), dtype=torch.float32),
    #         )

    #         log_factor = log(self.leakage_correction(x=batched_first_of_batch(x))) if norm_posterior else 0

    #         return masked_log_prob - log_factor

    # @torch.no_grad()
    # def leakage_correction(
    #     self,
    #     x: Tensor,
    #     num_rejection_samples: int = 10_000,
    #     force_update: bool = False,
    #     show_progress_bars: bool = False,
    #     rejection_sampling_batch_size: int = 10_000,
    # ) -> Tensor:
    #     r"""Return leakage correction factor for a leaky posterior density estimate.

    #     The factor is estimated from the acceptance probability during rejection
    #     sampling from the posterior.

    #     This is to avoid re-estimating the acceptance probability from scratch
    #     whenever `log_prob` is called and `norm_posterior=True`. Here, it
    #     is estimated only once for `self.default_x` and saved for later. We
    #     re-evaluate only whenever a new `x` is passed.

    #     Arguments:
    #         x: Conditioning context for posterior $p(\theta|x)$.
    #         num_rejection_samples: Number of samples used to estimate correction factor.
    #         force_update: Whether to force a reevaluation of the leakage correction even
    #             if the context `x` is the same as `self.default_x`. This is useful to
    #             enforce a new leakage estimate for rounds after the first (2, 3,..).
    #         show_progress_bars: Whether to show a progress bar during sampling.
    #         rejection_sampling_batch_size: Batch size for rejection sampling.

    #     Returns:
    #         Saved or newly-estimated correction factor (as a scalar `Tensor`).
    #     """

    #     def acceptance_at(x: Tensor) -> Tensor:
    #         return utils.sample_posterior_within_prior(
    #             self.net,
    #             self._prior,
    #             x.to(self._device),
    #             num_rejection_samples,
    #             show_progress_bars,
    #             sample_for_correction_factor=True,
    #             max_sampling_batch_size=rejection_sampling_batch_size,
    #         )[1]

    #     # Check if the provided x matches the default x (short-circuit on identity).
    #     is_new_x = self.default_x is None or (x is not self.default_x and (x != self.default_x).any())

    #     not_saved_at_default_x = self._leakage_density_correction_factor is None

    #     if is_new_x:  # Calculate at x; don't save.
    #         return acceptance_at(x)
    #     elif not_saved_at_default_x or force_update:  # Calculate at default_x; save.
    #         self._leakage_density_correction_factor = acceptance_at(self.default_x)

    #     return self._leakage_density_correction_factor  # type:ignore

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Samples are obtained either with rejection sampling or MCMC. Rejection sampling
        will be a lot faster if leakage is rather low. If leakage is high (e.g. over
        99%, which can happen in multi-round SNPE), MCMC can be faster than rejection
        sampling.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with_mcmc: Optional parameter to override `self.sample_with_mcmc`.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
        Returns:
            Samples from posterior.
        """

        num_samples = torch.Size(sample_shape).numel()

        self.net.eval()
        # Rejection sampling.
        samples, _ = utils.sample_posterior_within_prior(
            self.net,
            self._prior,
            x,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
        )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))