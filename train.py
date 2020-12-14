#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

sys.path.append("./")

import logging
import argparse
import numpy as np

import healpy as hp
from models.embedding import SphericalGraphCNN
from utils import create_mask as cm

import torch

from sbi import utils
from sbi.inference import SNPE


def train(data_dir, model_filename, sample_name, nside_max=128, r_outer=25, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("")
    logging.info("")
    logging.info("")
    logging.info("Creating estimator")
    logging.info("")

    # Get mask of central pixel for nside=1
    hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask=True, band_mask_range=0, mask_ring=True, inner=0, outer=r_outer)

    indexes_list = []
    masks_list = []

    assert (nside_max & (nside_max - 1) == 0) and nside_max != 0, "Invalid nside"

    nside_list = [int(nside_max / (2 ** i)) for i in np.arange(hp.nside2order(nside_max))]

    for nside in nside_list:
        hp_mask = hp.ud_grade(hp_mask_nside1, nside)
        masks_list.append(hp_mask)
        indexes_list.append(np.arange(hp.nside2npix(nside))[~hp_mask])

    # Embedding net
    sg_embed = SphericalGraphCNN(nside_list, indexes_list)

    # Priors hard-coded for now.
    prior = utils.BoxUniform(low=torch.tensor([0.001, 0.001, 10.0, 1.1, -10.0, 5.0, 0.1]), high=torch.tensor([0.5, 0.5, 20.0, 1.99, 1.99, 50.0, 4.99]))
    # Embedding net
    sg_embed = SphericalGraphCNN(nside_list, indexes_list)

    # Instantiate the neural density estimator
    neural_classifier = utils.posterior_nn(model="maf", embedding_net=sg_embed, hidden_features=50, num_transforms=4,)

    # Setup the inference procedure with the SNPE-C procedure
    inference_inst = SNPE(prior=prior, density_estimator=neural_classifier, show_progress_bars=True, logging_level="INFO", device=device.type)

    x_filename = "{}/samples/x_{}.npy".format(data_dir, sample_name)
    theta_filename = "{}/samples/theta_{}.npy".format(data_dir, sample_name)

    density_estimator = inference_inst.train(x=x_filename, theta=theta_filename, proposal=prior, training_batch_size=64, max_num_epochs=100)
    torch.save(density_estimator, "{}/models/{}.pt".format(data_dir, model_filename))


def parse_args():
    parser = argparse.ArgumentParser(description="High-level script for the training of the neural likelihood ratio estimators")

    # Main options
    parser.add_argument("--sample", type=str, help='Sample name, like "train".')
    parser.add_argument("--name", type=str, help="Model name. Defaults to the name of the method.")
    parser.add_argument(
        "--dir", type=str, default=".", help="Directory. Training data will be loaded from the data/samples subfolder, the model saved in the " "data/models subfolder.",
    )

    # Training option
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO,
    )
    logging.info("Hi!")

    args = parse_args()

    train(data_dir="{}/data/".format(args.dir), sample_name=args.sample, model_filename=args.name)

    logging.info("All done! Have a nice day!")
