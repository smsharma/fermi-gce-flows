# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import logging

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, NewType, Optional, Union, cast
from warnings import warn
import six
from collections import OrderedDict
import os

import torch
from torch import Tensor, ones, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.types import TorchModule
from sbi.utils import (
    check_estimator_arg,
    x_shape_from_simulation,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class PosteriorEstimatorNet(pl.LightningModule):
    """
    This is a wrapper class for the neural network defined by pyknos / nflows. It wraps
    the neural network into a pytorch_lightning module.
    """

    def __init__(self, net, proposal, loss, lr, calibration_kernel):
        """
        Initialize the posterior estimation net.
        The reason that this is a dict: when listing all arguments separately, pytorch-
        lightning breaks when one calls `load_from_checkpoint` if the arguments have
        different types.
        Args:
            args: Dict containing `net`, `proposal`, `loss`, lr`, `calibration_kernel`.
                See below for further explanation.
            `net`: Neural density estimator.
            `proposal`: Proposal distribution.
            `loss`: Loss function.
            `lr`: Learning rate.
            `calibration_kernel`: Calibration kernel.
        """

        super().__init__()

        self.save_hyperparameters('lr')

        self.net = net
        self.proposal = proposal
        self.loss = loss
        self.lr = lr
        self.calibration_kernel = calibration_kernel

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.net.parameters()), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        theta, x = batch
        loss = torch.mean(
            self.loss(theta, x, self.proposal, self.calibration_kernel)
        )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        theta, x = batch
        loss = torch.mean(
            self.loss(theta, x, self.proposal, self.calibration_kernel)
        )
        self.log('val_loss', loss)


class PosteriorEstimator(NeuralInference, ABC):
    def __init__(self, prior, density_estimator: Union[str, Callable] = "maf", device: str = "cpu", logging_level: Union[int, str] = "WARNING", summary_writer: Optional[SummaryWriter] = None, show_progress_bars: bool = True, **unused_args):
        """Base class for Sequential Neural Posterior Estimation methods.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            **unused_args,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = utils.posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        # Extra SNPE-specific fields summary_writer.
        self._summary.update({"rejection_sampling_acceptance_rates": []})  # type:ignore

    def train(
        self,
        x,
        theta,
        proposal,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
    ) -> DirectPosterior:
        r"""
        Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
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
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # Load data from most recent round.
        theta = self.load_and_check(theta, memmap=False)
        x = self.load_and_check(x, memmap=True)

        data = OrderedDict()
        data["theta"] = theta
        data["x"] = x
        dataset = self.make_dataset(data)

        train_loader, val_loader = self.make_dataloaders(dataset, validation_fraction, training_batch_size)

        num_z_score = 10000  # Z-score using a limited sample for memory reasons
        theta_z_score, x_z_score = train_loader.dataset[:num_z_score]

        logging.info("Z-scoring using {} random training samples for x".format(num_z_score))

        # Call the `self._build_neural_net` which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        self._neural_net = self._build_neural_net(theta_z_score, torch.Tensor(x))
        self._x_shape = x_shape_from_simulation(torch.Tensor(x))

        max_num_epochs=cast(int, max_num_epochs)

        self._model = PosteriorEstimatorNet(
            self._neural_net,
            proposal,
            self._loss,
            learning_rate,
            calibration_kernel
        )

        # Hard code not to save anything
        model_checkpoint = ModelCheckpoint(monitor='val_loss', dirpath="./data/models/", filename="{epoch:02d}-{val_loss:.2f}")
        checkpoint_callback = model_checkpoint

        early_stop_callback = EarlyStopping(monitor='val_loss', patience=stop_after_epochs)
        
        trainer = pl.Trainer(
            logger=self._summary_writer,
            callbacks=[early_stop_callback,checkpoint_callback],
            gradient_clip_val=clip_max_norm,
            max_epochs=max_num_epochs,
            progress_bar_refresh_rate=self._show_progress_bars,
            deterministic=False,
            gpus=1
        )

        trainer.fit(self._model, train_loader, val_loader)

        # Load the model that had the best validation log-probability
        self._best_model = PosteriorEstimatorNet.load_from_checkpoint(
            checkpoint_path=model_checkpoint.best_model_path,
            net=self._neural_net,
            proposal=proposal,
            loss=self._loss,
            calibration_kernel=calibration_kernel,
        )

        # Return the posterior net corresponding to the best model
        return self._best_model.net

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> DirectPosterior:
        r"""
        Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:

        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `max_sampling_batch_size` to set the batch size for drawing new
                samples from the candidate distribution, e.g., the posterior. Larger
                batch size speeds up sampling.
            sample_with_mcmc: Whether to sample with MCMC. MCMC can be used to deal
                with high leakage.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior` will
                draw init locations from prior, whereas `sir` will use
                Sequential-Importance-Resampling using `init_strategy_num_candidates`
                to find init locations.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """

        if density_estimator is None:
            density_estimator = self._neural_net

        self._posterior = DirectPosterior(
            method_family="snpe",
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self._x_shape,
            rejection_sampling_parameters=rejection_sampling_parameters,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            device=self._device,
        )

        # Posterior in eval mode
        self._posterior.net.eval()

        return deepcopy(self._posterior)

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
        """

        # Use posterior log prob
        log_prob = self._neural_net.log_prob(theta, x)

        return -(calibration_kernel(x) * log_prob)

    def make_dataset(self, data):
        data_arrays = []
        data_labels = []
        for key, value in six.iteritems(data):
            data_labels.append(key)
            data_arrays.append(value)
        dataset = NumpyDataset(*data_arrays, dtype=torch.float)  # Should maybe mod dtype
        return dataset

    def make_dataloaders(self, dataset, validation_split, batch_size, seed=None):
        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=8,
            )  ## Run on GPU
            val_loader = None
            num_validation_examples = 0
        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

            n_samples = len(dataset)
            indices = list(range(n_samples))
            split = int(np.floor(validation_split * n_samples))
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=8,
            )  ## Run on GPU
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=8,
            )  ## Run on GPU

        return train_loader, val_loader

    def load_and_check(self, filename, memmap=False):
        # Don't load image files > 1 GB into memory
        if memmap and os.stat(filename).st_size > 1.0 * 1024 ** 3:
            data = np.load(filename, mmap_mode="c")
        else:
            data = np.load(filename)
        return data


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, dtype=torch.float):
        self.dtype = dtype
        self.memmap = []
        self.data = []
        self.n = None

        for array in arrays:
            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n
