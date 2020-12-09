# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from math import pi
from typing import Any, Callable, Optional, Union

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from pyknos.nflows.transforms import CompositeTransform
from torch import Tensor, eye, ones
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import (
    batched_mixture_mv,
    batched_mixture_vmv,
    clamp_and_warn,
    del_entries,
    repeat_rows,
)


class SNPE_C(PosteriorEstimator):
    def __init__(
        self, prior, density_estimator: Union[str, Callable] = "maf", device: str = "cpu", logging_level: Union[int, str] = "WARNING", summary_writer: Optional[TensorboardSummaryWriter] = None, show_progress_bars: bool = True, **unused_args,
    ):
        r"""SNPE-C / APT [1].

        [1] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        This class implements two loss variants of SNPE-C: the non-atomic and the atomic
        version. The atomic loss of SNPE-C can be used for any density estimator,
        i.e. also for normalizing flows. However, it suffers from leakage issues. On
        the other hand, the non-atomic loss can only be used only if the proposal
        distribution is a mixture of Gaussians, the density estimator is a mixture of
        Gaussians, and the prior is either Gaussian or Uniform. It does not suffer from
        leakage issues. At the beginning of each round, we print whether the non-atomic
        or the atomic version is used.

        In this codebase, we will automatically switch to the non-atomic loss if the
        following criteria are fulfilled:<br/>
        - proposal is a `DirectPosterior` with density_estimator `mdn`, as built
            with `utils.sbi.posterior_nn()`.<br/>
        - the density estimator is a `mdn`, as built with
            `utils.sbi.posterior_nn()`.<br/>
        - `isinstance(prior, MultivariateNormal)` (from `torch.distributions`) or
            `isinstance(prior, sbi.utils.BoxUniform)`

        Note that custom implementations of any of these densities (or estimators) will
        not trigger the non-atomic loss, and the algorithm will fall back onto using
        the atomic loss.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__", "unused_args"))
        super().__init__(**kwargs, **unused_args)

    def train(self, x, theta, proposal, num_atoms: int = 10, training_batch_size: int = 50, learning_rate: float = 5e-4, validation_fraction: float = 0.1, stop_after_epochs: int = 20, max_num_epochs: Optional[int] = None, clip_max_norm: Optional[float] = 5.0, calibration_kernel: Optional[Callable] = None, exclude_invalid_x: bool = True, discard_prior_samples: bool = False, use_combined_loss: bool = False, retrain_from_scratch_each_round: bool = False, show_train_summary: bool = False,) -> DirectPosterior:
        r"""
        Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss
        kwargs = del_entries(locals(), entries=("self", "__class__", "num_atoms", "use_combined_loss"))

        # Hard-coded single round
        self._data_round_index.append(0)
        self._round = max(self._data_round_index)
        return super().train(**kwargs)

    def _log_prob_proposal_posterior(self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any]) -> Tensor:
        """
        Return the log-probability of the proposal posterior.

        If the proposal is a MoG, the density estimator is a MoG, and the prior is
        either Gaussian or uniform, we use non-atomic loss. Else, use atomic loss (which
        suffers from leakage).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.

        Returns: Log-probability of the proposal posterior.
        """

        return self._log_prob_proposal_posterior_atomic(theta, x, masks)

    def _log_prob_proposal_posterior_atomic(self, theta: Tensor, x: Tensor, masks: Tensor):
        """
        Return log probability of the proposal posterior for atomic proposals.

        We have two main options when evaluating the proposal posterior.
            (1) Generate atoms from the proposal prior.
            (2) Generate atoms from a more targeted distribution, such as the most
                recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly-initialized neural density
        estimator.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.

        Returns:
            Log-probability of the proposal posterior.
        """

        batch_size = theta.shape[0]

        num_atoms = clamp_and_warn("num_atoms", self._num_atoms, min_val=2, max_val=batch_size)

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(batch_size * num_atoms, -1)

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._neural_net.log_prob(atomic_theta, repeated_x)
        self._assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        self._assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(unnormalized_log_prob, dim=-1)
        self._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        return log_prob_proposal_posterior
