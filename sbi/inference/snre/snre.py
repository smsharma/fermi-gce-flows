import logging

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, NewType, Optional, Union, cast
from warnings import warn
import six
from collections import OrderedDict
import os
from pathlib import Path

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, eye, ones, optim
from torch import Tensor, nn, ones

import torch
from torch import Tensor, ones, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

from abc import ABC
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import TensorboardSummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sbi.utils import del_entries

from sbi import utils as utils
from sbi.inference import NeuralInference, EstimatorNet
from sbi.inference.posteriors.ratio_based_posterior import RatioBasedPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.types import TorchModule
from sbi.utils import x_shape_from_simulation

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import mlflow.pytorch


class RatioEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior,
        classifier: Union[str, Callable] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args
    ):
        r"""Sequential Neural Ratio Estimation.

        We implement two inference methods in the respective subclasses.

        - SNRE_A / AALR is limited to `num_atoms=2`, but allows for density evaluation
          when training for one round.
        - SNRE_B / SRE can use more than two atoms, potentially boosting performance,
          but allows for posterior evaluation **only up to a normalizing constant**,
          even when training only one round.

        Args:
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations (theta, x), which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.
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
            **unused_args
        )

        self._build_neural_net = classifier

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
    ) -> RatioBasedPosterior:

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
        theta_z_score, x_z_score, x_aux_z_score = train_loader.dataset[:num_z_score]

        logging.info("Z-scoring using {} random training samples for x".format(num_z_score))

        x_and_aux_z_score = torch.cat([x_z_score, x_aux_z_score], -1)

        # Call the `self._build_neural_net` which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        self.neural_net = self._build_neural_net(theta_z_score, x_and_aux_z_score)
        self.x_shape = x_shape_from_simulation(x_and_aux_z_score)

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
            gpus=None,  # Hard-coded
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
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> RatioBasedPosterior:
        r"""
        Build posterior from the neural density estimator.

        SNRE trains a neural network to approximate likelihood ratios, which in turn
        can be used obtain an unnormalized posterior
        $p(\theta|x) \propto p(x|\theta) \cdot p(\theta)$. The posterior returned here
        wraps the trained network such that one can directly evaluate the unnormalized
        posterior log-probability $p(\theta|x) \propto p(x|\theta) \cdot p(\theta)$ and
        draw samples from the posterior with MCMC. Note that, in the case of
        single-round SNRE_A / AALR, it is possible to evaluate the log-probability of
        the **normalized** posterior, but sampling still requires MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
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
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        if density_estimator is None:
            density_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device

        self._posterior = RatioBasedPosterior(
            method_family=self.__class__.__name__.lower(),
            neural_net=density_estimator,
            prior=self._prior,
            x_shape=self._x_shape,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            device=device,
        )

        self._posterior._num_trained_rounds = self._round + 1

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        return deepcopy(self._posterior)

    def classifier_logits(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """Return logits obtained through classifier forward pass.

        The logits are obtained from atomic sets of (theta,x) pairs.
        """
        batch_size = theta.shape[0]
        repeated_x = utils.repeat_rows(x, num_atoms)
        
        # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

        contrasting_theta = theta[choices]

        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        return self.neural_net(repeated_x, atomic_theta)

    def loss(self, theta: Tensor, x: Tensor, proposal: Optional[Any],) -> Tensor:
        """
        Returns the binary cross-entropy loss for the trained classifier.

        The classifier takes as input a $(\theta,x)$ pair. It is trained to predict 1
        if the pair was sampled from the joint $p(\theta,x)$, and to predict 0 if the
        pair was sampled from the marginals $p(\theta)p(x)$.
        """

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        num_atoms = 2

        logits = self.classifier_logits(theta, x, num_atoms)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * batch_size)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        return nn.BCELoss()(likelihood, labels)