from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
import argparse
import logging

import numpy as np
import healpy as hp
from tqdm.auto import tqdm
import torch

logger = logging.getLogger(__name__)
sys.path.append("./")
sys.path.append("../")
sys.path.append("../sbi/")

from sbi import utils
from simulations.wrapper import simulator
from utils import create_mask as cm
from models.psf import KingPSF


def simulate(n=10000, r_outer=25, nside_max=128, psf="king"):

    hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask=True, band_mask_range=0, mask_ring=True, inner=0, outer=r_outer)

    indexes_list = []
    masks_list = []

    assert (nside_max & (nside_max - 1) == 0) and nside_max != 0, "Invalid nside"

    nside_list = [int(nside_max / (2 ** i)) for i in np.arange(hp.nside2order(nside_max))]

    for nside in nside_list:
        hp_mask = hp.ud_grade(hp_mask_nside1, nside)
        masks_list.append(hp_mask)
        indexes_list.append(np.arange(hp.nside2npix(nside))[~hp_mask])

    temp_gce = np.load("data/fermi_data/template_gce.npy")

    if psf == "king":
        kp = KingPSF()

    else:
        raise NotImplementedError

    logger.info("Generating training data with %s images", n)

    prior = utils.BoxUniform(low=torch.tensor([0.5, 10.0, 1.1, -10.0, 5.0, 0.1]), high=torch.tensor([3.0, 20.0, 1.99, 1.99, 50.0, 4.99]))
    thetas = prior.sample((n,))

    # Generate images
    logger.info("Generating %s maps", n)

    x = [simulator(theta.detach().numpy(), masks_list[0], temp_gce, kp.psf_fermi_r) for theta in tqdm(thetas)]

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

    parser = argparse.ArgumentParser(description="Main high-level script that starts the GCE simulations")

    parser.add_argument(
        "-n", type=int, default=10000, help="Number of samples to generate. Default is 10k.",
    )
    parser.add_argument("--name", type=str, default=None, help='Sample name, like "train" or "test".')
    parser.add_argument("--dir", type=str, default=".", help="Base directory. Results will be saved in the data/samples subfolder.")
    parser.add_argument("--debug", action="store_true", help="Prints debug output.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")

    name = "train" if args.name is None else args.name
    results = simulate(args.n)
    save(args.dir, name, results)

    logger.info("All done! Have a nice day!")
