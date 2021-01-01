# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from warnings import warn

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils import get_log_root
from sbi.utils.torchutils import process_device

from pytorch_lightning.loggers import TensorBoardLogger


class NeuralInference(ABC):
    """Abstract base class for neural inference methods."""

    def __init__(
        self, prior, device: str = "cpu", logging_level: Union[int, str] = "WARNING", summary_writer: Optional[SummaryWriter] = None, show_progress_bars: bool = True, **unused_args,
    ):
        r"""
        Base class for inference methods.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            device: torch device on which to compute, e.g. gpu or cpu.
            logging_level: Minimum severity of messages to log. One of the strings
               "INFO", "WARNING", "DEBUG", "ERROR" and "CRITICAL".
            summary_writer: A `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.
        """

        self._device = process_device(device)

        if unused_args:
            warn(f"You passed some keyword arguments that will not be used. " f"Specifically, the unused arguments are: {list(unused_args.keys())}. " f"These arguments might have been supported in sbi " f"versions <0.14.0. Since 0.14.0, the API was changed. Please consult " f"the corresponding pull request on github: " f"https://github.com/mackelab/sbi/pull/378 and tutorials: " f"https://www.mackelab.org/sbi/tutorial/02_flexible_interface/ for " f"further information.",)

        self._prior = prior
        self._posterior = None
        self._neural_net = None
        self._x_shape = None

        self._show_progress_bars = show_progress_bars

        # Initialize roundwise (theta, x, prior_masks) for storage of parameters,
        # simulations and masks indicating if simulations came from prior.
        self._theta_roundwise, self._x_roundwise, self._prior_masks = [], [], []
        self._model_bank = []

        # Initialize list that indicates the round from which simulations were drawn.
        self._data_round_index = []

        self._round = 0

        # XXX We could instantiate here the Posterior for all children. Two problems:
        #     1. We must dispatch to right PotentialProvider for mcmc based on name
        #     2. `method_family` cannot be resolved only from `self.__class__.__name__`,
        #         since SRE, AALR demand different handling but are both in SRE class.

        self.summary_writer = self._default_summary_writer() if summary_writer is None else summary_writer

    def __call__(self, unused_args):
        """
        f"Deprecated. The inference object is no longer callable as of `sbi` v0.14.0. "
        f"Please consult our website for a tutorial on how to adapt your code: "
        f"https://www.mackelab.org/sbi/tutorial/02_flexible_interface/ . For "
        f"further information, see the corresponding pull request on github:
        f"https://github.com/mackelab/sbi/pull/378.",
        """
        raise NameError(f"The inference object is no longer callable as of `sbi` v0.14.0. " f"Please consult the release notes for a tutorial on how to adapt your " f"code: https://github.com/mackelab/sbi/releases/tag/v0.14.0" f"For more information, visit our website: " f"https://www.mackelab.org/sbi/tutorial/02_flexible_interface/ or " f"see the corresponding pull request on github: " f"https://github.com/mackelab/sbi/pull/378.",)

    def provide_presimulated(self, theta: Tensor, x: Tensor, from_round: int = 0) -> None:
        r"""
        Deprecated since sbi 0.14.0.

        Instead of using this, please use `.append_simulations()`. Please consult
        release notes to see how you can update your code:
        https://github.com/mackelab/sbi/releases/tag/v0.14.0
        More information can be found under the corresponding pull request on github:
        https://github.com/mackelab/sbi/pull/378
        and tutorials:
        https://www.mackelab.org/sbi/tutorial/02_flexible_interface/

        Provide external $\theta$ and $x$ to be used for training later on.

        Args:
            theta: Parameter sets used to generate presimulated data.
            x: Simulation outputs of presimulated data.
            from_round: Which round the data was simulated from. `from_round=0` means
                that the data came from the first round, i.e. the prior.
        """
        raise NameError(f"Deprecated since sbi 0.14.0. " f"Instead of using this, please use `.append_simulations()`. Please " f"consult release notes to see how you can update your code: " f"https://github.com/mackelab/sbi/releases/tag/v0.14.0" f"More information can be found under the corresponding pull request on " f"github: " f"https://github.com/mackelab/sbi/pull/378" f"and tutorials: " f"https://www.mackelab.org/sbi/tutorial/02_flexible_interface/",)

    @abstractmethod
    def train(self, training_batch_size: int = 50, learning_rate: float = 5e-4, validation_fraction: float = 0.1, stop_after_epochs: int = 20, max_num_epochs: Optional[int] = None, clip_max_norm: Optional[float] = 5.0, calibration_kernel: Optional[Callable] = None, exclude_invalid_x: bool = True, discard_prior_samples: bool = False, retrain_from_scratch_each_round: bool = False, show_train_summary: bool = False,) -> NeuralPosterior:
        raise NotImplementedError

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            neural_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def _default_summary_writer(self) -> TensorBoardLogger:
        """Return summary writer logging to method- and simulator-specific directory."""
        method = self.__class__.__name__
        logdir = Path(get_log_root(), method, datetime.now().isoformat().replace(":", "_"),)
        return TensorBoardLogger(logdir)

    @staticmethod
    def _ensure_list(num_simulations_per_round: Union[List[int], int], num_rounds: int) -> List[int]:
        """Return `num_simulations_per_round` as a list of length `num_rounds`.
        """
        try:
            assert len(num_simulations_per_round) == num_rounds, "Please provide a list with number of simulations per round for each " "round, or a single integer to be used for all rounds."
        except TypeError:
            num_simulations_per_round: List = [num_simulations_per_round] * num_rounds

        return cast(list, num_simulations_per_round)

    # @staticmethod
    # def _describe_round(round_: int, summary: Dict[str, list]) -> str:
    #     epochs = summary["epochs"][-1]
    #     best_validation_log_probs = summary["best_validation_log_probs"][-1]

    #     description = f"""
    #     -------------------------
    #     ||||| ROUND {round_ + 1} STATS |||||:
    #     -------------------------
    #     Epochs trained: {epochs}
    #     Best validation performance: {best_validation_log_probs:.4f}
    #     -------------------------
    #     """

    #     return description

    @staticmethod
    def _maybe_show_progress(show=bool, epoch=int) -> None:
        if show:
            # end="\r" deletes the print statement when a new one appears.
            # https://stackoverflow.com/questions/3419984/
            print("Training neural network. Epochs trained: ", epoch, end="\r")

    def _report_convergence_at_end(self, epoch: int, stop_after_epochs: int, max_num_epochs: int) -> None:
        if self._converged(epoch, stop_after_epochs):
            print(f"Neural network successfully converged after {epoch} epochs.")
        elif max_num_epochs == epoch:
            warn("Maximum number of epochs `max_num_epochs={max_num_epochs}` reached," "but network has not yet fully converged. Consider increasing it.")

    @staticmethod
    def _assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
        """Raise if tensor quantity contains any NaN or Inf element."""

        msg = f"NaN/Inf present in {description}."
        assert torch.isfinite(quantity).all(), msg
