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
from pathlib import Path

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
from sbi.inference import NeuralInference, EstimatorNet
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.types import TorchModule
from sbi.utils import x_shape_from_simulation,

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import mlflow.pytorch


class PosteriorEstimator(NeuralInference, ABC):
    def __init__(self, prior, density_estimator: Union[str, Callable] = "maf", logging_level: Union[int, str] = "WARNING", summary_writer: Optional[SummaryWriter] = None, show_progress_bars: bool = True, **unused_args):

        super().__init__(
            prior=prior,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            **unused_args,
        )

        self._build_neural_net = density_estimator

    def train(
        self,
        x,
        x_aux,
        theta,
        proposal,
        optimizer=optim.AdamW,
        optimizer_kwargs=None,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        training_batch_size: int = 50,
        initial_lr: float = 1e-3,
        validation_fraction: float = 0.25,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 1.0,
    ) -> DirectPosterior:

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        scheduler_kwargs = {'T_max':max_num_epochs} if scheduler_kwargs is None else scheduler_kwargs


        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # Load data
        theta = self.load_and_check(theta, memmap=False)
        x = self.load_and_check(x, memmap=True)
        x_aux = self.load_and_check(x_aux, memmap=False)

        data = OrderedDict()
        data["theta"] = theta
        data["x"] = x
        data["x_aux"] = x_aux
        dataset = self.make_dataset(data)

        train_loader, val_loader = self.make_dataloaders(dataset, validation_fraction, training_batch_size)

        num_z_score = 50000  # Z-score using a limited random sample for memory reasons
        theta_z_score, x_z_score = train_loader.dataset[:num_z_score]

        logging.info("Z-scoring using {} random training samples for x".format(num_z_score))

        # Call the `self._build_neural_net` which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        self.neural_net = self._build_neural_net(theta_z_score, x_z_score)
        self.x_shape = x_shape_from_simulation(x_z_score)

        max_num_epochs=cast(int, max_num_epochs)

        self.model = EstimatorNet(
            net=self.neural_net,
            proposal=proposal,
            loss=self.loss,
            initial_lr=initial_lr, 
            optimizer=optimizer, 
            optimizer_kwargs=optimizer_kwargs, 
            scheduler=scheduler, 
            scheduler_kwargs=scheduler_kwargs,
        )

        checkpoint_path = "{}/{}/{}/artifacts/checkpoints/".format(self.summary_writer.save_dir, self.summary_writer.experiment_id, self.summary_writer.run_id)
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)

        model_checkpoint = ModelCheckpoint(monitor='val_loss', dirpath=checkpoint_path, filename="{epoch:02d}-{val_loss:.2f}", period=5, save_top_k=3)
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
        mlflow.pytorch.autolog(log_models=False)

        # Train the model
        with mlflow.start_run(run_id=self.summary_writer.run_id) as run:
            trainer.fit(self.model, train_loader, val_loader)

        # Load the model that had the best validation log-probability
        self.best_model = EstimatorNet.load_from_checkpoint(
            checkpoint_path=model_checkpoint.best_model_path,
            net=self.neural_net,
            proposal=proposal,
            loss=self.loss,
            initial_lr=initial_lr, 
            optimizer=optimizer, 
            optimizer_kwargs=optimizer_kwargs, 
            scheduler=scheduler, 
            scheduler_kwargs=scheduler_kwargs,
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
        x_aux: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
 
        # Use posterior log prob
        log_prob = self.neural_net.log_prob(theta, x, x_aux)

        return -log_prob
