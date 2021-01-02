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
    x_shape_from_simulation,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import mlflow.pytorch

class PosteriorEstimatorNet(pl.LightningModule):
    """
    This is a wrapper class for the neural network defined by pyknos / nflows. It wraps
    the neural network into a pytorch_lightning module.
    """

    def __init__(self, net, proposal, loss, initial_lr, optimizer, optimizer_kwargs, scheduler, scheduler_kwargs, calibration_kernel):
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

        self.save_hyperparameters('initial_lr')

        self.net = net
        self.proposal = proposal
        self.loss = loss

        self.initial_lr = initial_lr
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.calibration_kernel = calibration_kernel

    def configure_optimizers(self):
        optimizer = self.optimizer(list(self.net.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        theta, x = batch
        loss = torch.mean(
            self.loss(theta, x, self.proposal, self.calibration_kernel)
        )
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta, x = batch
        loss = torch.mean(
            self.loss(theta, x, self.proposal, self.calibration_kernel)
        )
        self.log('val_loss', loss)


class PosteriorEstimator(NeuralInference, ABC):
    def __init__(self, prior, density_estimator: Union[str, Callable] = "maf", device: str = "cpu", logging_level: Union[int, str] = "WARNING", summary_writer: Optional[SummaryWriter] = None, show_progress_bars: bool = True, **unused_args):

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
        # `_build_neural_net`
        self._build_neural_net = density_estimator

    def train(
        self,
        x,
        theta,
        proposal,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        training_batch_size: int = 50,
        initial_lr: float = 1e-3,
        validation_fraction: float = 0.25,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 1.0,
        calibration_kernel: Optional[Callable] = None,
    ) -> DirectPosterior:

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        scheduler_kwargs = {'T_max':max_num_epochs} if scheduler_kwargs is None else scheduler_kwargs

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # Load data
        theta = self.load_and_check(theta, memmap=False)
        x = self.load_and_check(x, memmap=True)

        data = OrderedDict()
        data["theta"] = theta
        data["x"] = x
        dataset = self.make_dataset(data)

        train_loader, val_loader = self.make_dataloaders(dataset, validation_fraction, training_batch_size)

        num_z_score = 10000  # Z-score using a limited random sample for memory reasons
        theta_z_score, x_z_score = train_loader.dataset[:num_z_score]

        logging.info("Z-scoring using {} random training samples for x".format(num_z_score))

        # Call the `self._build_neural_net` which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        self.neural_net = self._build_neural_net(theta_z_score, torch.Tensor(x))
        self.x_shape = x_shape_from_simulation(torch.Tensor(x))

        max_num_epochs=cast(int, max_num_epochs)

        self.model = PosteriorEstimatorNet(
            net=self.neural_net,
            proposal=proposal,
            loss=self.loss,
            initial_lr=initial_lr, 
            optimizer=optimizer, 
            optimizer_kwargs=optimizer_kwargs, 
            scheduler=scheduler, 
            scheduler_kwargs=scheduler_kwargs,
            calibration_kernel=calibration_kernel
        )

        model_checkpoint = ModelCheckpoint(monitor='val_loss', dirpath="./data/models/", filename="{epoch:02d}-{val_loss:.2f}")
        checkpoint_callback = model_checkpoint
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=stop_after_epochs)        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            logger=self.summary_writer,
            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
            gradient_clip_val=clip_max_norm,
            max_epochs=max_num_epochs,
            progress_bar_refresh_rate=self._show_progress_bars,
            deterministic=False,
            gpus=[0],  # Hard-coded
            num_sanity_val_steps=10,
        )

        # Auto log all MLflow entities
        mlflow.set_tracking_uri(self.summary_writer._tracking_uri)
        mlflow.pytorch.autolog()

        # Train the model
        with mlflow.start_run(run_id=self.summary_writer.run_id) as run:
            trainer.fit(self.model, train_loader, val_loader)

        # Load the model that had the best validation log-probability
        self.best_model = PosteriorEstimatorNet.load_from_checkpoint(
            checkpoint_path=model_checkpoint.best_model_path,
            net=self.neural_net,
            proposal=proposal,
            loss=self.loss,
            initial_lr=initial_lr, 
            optimizer=optimizer, 
            optimizer_kwargs=optimizer_kwargs, 
            scheduler=scheduler, 
            scheduler_kwargs=scheduler_kwargs,
            calibration_kernel=calibration_kernel
        )

        # Return the posterior net corresponding to the best model
        return self.best_model.net

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> DirectPosterior:

        if density_estimator is None:
            density_estimator = self.neural_net

        self._posterior = DirectPosterior(
            method_family="snpe",
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self.x_shape,
            rejection_sampling_parameters=rejection_sampling_parameters,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            device=self._device,
        )

        # Posterior in eval mode
        self._posterior.net.eval()

        return deepcopy(self._posterior)

    def loss(
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
        log_prob = self.neural_net.log_prob(theta, x)

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
