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
from sbi.utils.torchutils import process_device, seed_worker

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import six
import os

import pytorch_lightning as pl


class EstimatorNet(pl.LightningModule):
    """
    This is a wrapper class for the neural network defined by pyknos / nflows. It wraps
    the neural network into a pytorch_lightning module.
    """

    def __init__(self, net, proposal, loss, initial_lr, optimizer, optimizer_kwargs, scheduler, scheduler_kwargs, summary=False):

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
        self.summary = summary

    def configure_optimizers(self):
        optimizer = self.optimizer(list(self.net.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        theta, x, x_aux = batch
        x_and_aux = torch.cat([x, x_aux], -1)
        if self.summary:
            x_and_aux = torch.squeeze(x_and_aux, 1)
        loss = torch.mean(self.loss(theta, x_and_aux, self.proposal))
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta, x, x_aux = batch
        x_and_aux = torch.cat([x, x_aux], -1)
        if self.summary:
            x_and_aux = torch.squeeze(x_and_aux, 1)
        loss = torch.mean(self.loss(theta, x_and_aux, self.proposal))
        self.log('val_loss', loss, on_epoch=True)

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
        self.summary_writer =  summary_writer

    @staticmethod
    def make_dataset(data):
        data_arrays = []
        data_labels = []
        for key, value in six.iteritems(data):
            data_labels.append(key)
            data_arrays.append(value)
        dataset = NumpyDataset(*data_arrays, dtype=torch.float)  # Should maybe mod dtype
        return dataset

    @staticmethod
    def make_datasets(data):
        datasets = []
        for i in range(len(data[next(iter(data))])):
            data_arrays = []
            for key, value in six.iteritems(data):
                data_arrays.append(value[i])
            datasets.append(NumpyDataset(*data_arrays, dtype=torch.float))
        return ConcatDataset(datasets)

    @staticmethod
    def make_dataloaders(dataset, validation_split, batch_size, num_workers=16, pin_memory=True, seed=None):
        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=num_workers,
            ) 
            val_loader = None
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
                pin_memory=pin_memory,
                num_workers=num_workers,
                # worker_init_fn=seed_worker
            ) 
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                # worker_init_fn=seed_worker
            )  

        return train_loader, val_loader

    @staticmethod
    def load_and_check(filename, memmap=False):
        # Don't load image files > 1 GB into memory
        # if memmap and os.stat(filename).st_size > 1.0 * 1024 ** 3:
        if memmap:
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