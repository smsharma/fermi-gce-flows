#! /usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from simulations.simulate_ps import SimulateMap
from utils import create_mask as cm
import healpy as hp

import torch


import sys, os
import argparse
import logging

logger = logging.getLogger(__name__)
sys.path.append("./")

from sbi import utils
from tqdm import *

from simulations.wrapper import simulator

from utils.psf_correction import PSFCorrection
from utils.utils import make_dirs
from models.psf import KingPSF


def simulate_train(n=10000):

    hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask=True, band_mask_range=0, mask_ring=True, inner=0, outer=25)

    indexes_list = []
    masks_list = []

    nside_list = [128, 64, 32, 16, 8, 4, 2]

    for nside in nside_list:
        hp_mask = hp.ud_grade(hp_mask_nside1, nside)
        masks_list.append(hp_mask)
        indexes_list.append(np.arange(hp.nside2npix(nside))[~hp_mask])

    kp = KingPSF()
    temp_gce = np.load("data/fermi_data/template_gce.npy")

    pc_inst = PSFCorrection(delay_compute=True)
    pc_inst.psf_r_func = lambda r: kp.psf_fermi_r(r)

    logger.info("Generating training data with %s images", n)

    prior = utils.BoxUniform(low=torch.tensor([0.5, 10.0, 1.1, -10.0, 5.0, 0.1]), high=torch.tensor([3.0, 20.0, 1.9, 1.9, 50.0, 4.99]))

    thetas = prior.sample((n,))

    # Samples from numerator
    logger.info("Generating %s images", n)
    x = [simulator(theta.detach().numpy(), masks_list[0], temp_gce, pc_inst.psf_r_func) for theta in tqdm(thetas)]
    results = {}
    results["theta"] = thetas
    results["x"] = x

    return results


def save(data_dir, name, data):
    logger.info("Saving results with name %s", name)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/data".format(data_dir)):
        os.mkdir("{}/data".format(data_dir))
    if not os.path.exists("{}/data/samples".format(data_dir)):
        os.mkdir("{}/data/samples".format(data_dir))

    for key, value in data.items():
        np.save("{}/data/samples/{}_{}.npy".format(data_dir, key, name), value)


def parse_args():
    parser = argparse.ArgumentParser(description="Main high-level script that starts the strong lensing simulations")

    parser.add_argument(
        "-n", type=int, default=10000, help="Number of samples to generate. Default is 10k.",
    )

    parser.add_argument("--name", type=str, default=None, help='Sample name, like "train" or "test".')

    parser.add_argument(
        "--dir", type=str, default=".", help="Base directory. Results will be saved in the data/samples subfolder.",
    )
    parser.add_argument("--debug", action="store_true", help="Prints debug output.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")

    name = "train" if args.name is None else args.name
    results = simulate_train(args.n)
    save(args.dir, name, results)

    logger.info("All done! Have a nice day!")
