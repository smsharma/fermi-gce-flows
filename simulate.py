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

from sbi import utils
from simulations.wrapper import simulator
from utils import create_mask as cm
from models.psf import KingPSF


def simulate(n=10000, r_outer=25, nside=128, psf="king", dif="ModelO"):
    """ High-level simulation script
    """

    # Get mask of central pixel for nside=1
    hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask=True, band_mask_range=0, mask_ring=True, inner=0, outer=r_outer)

    # Get mask corresponding to nside=128
    mask_sim = hp.ud_grade(hp_mask_nside1, nside)

    # Get ROI mask
    ps_mask = np.load("data/mask_3fgl_0p8deg.npy")
    mask_roi = cm.make_mask_total(nside=nside, band_mask = True, band_mask_range=2, mask_ring=True, inner=0, outer=25, custom_mask=ps_mask)

    # Load templates
    temp_gce = np.load("data/fermi_data/template_gce.npy")
    temp_dif = np.load("data/fermi_data/template_dif.npy")
    temp_psc = np.load("data/fermi_data/template_psc.npy")
    temp_iso = np.load("data/fermi_data/template_iso.npy")
    temp_dsk = np.load("data/fermi_data/template_dsk.npy")
    temp_bub = np.load("data/fermi_data/template_bub.npy")

    temp_mO_pibrem = np.load('data/fermi_data/ModelO_r25_q1_pibrem.npy')
    temp_mO_ics = np.load('data/fermi_data/ModelO_r25_q1_ics.npy')

    # King PSF hard-coded for now
    if psf == "king":
        kp = KingPSF()
    else:
        raise NotImplementedError

    logger.info("Generating training data with %s images", n)

    # Store and return
    results = {}

    # Poiss and PS templates

    if dif == "ModelO":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_mO_pibrem, temp_mO_ics]
        # iso, bub, psc, dif_pibrem, dif_ics
        prior_poiss = [[0.001, 0.001, 0.001, 6., 1.], [1.5, 1.5, 1.5, 12., 6.]]
    elif dif == "p6v11":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_dif]
        # iso, bub, psc, dif
        prior_poiss = [[0.001, 0.001, 0.001, 11.], [1.5, 1.5, 1.5, 16.]]
    else:
        raise NotImplementedError

    temps_ps = [temp_gce, temp_dsk]

    # gce, dsk
    prior_ps = [[0.001, 10.0, 1.1, -10.0, 5.0, 0.1, 0.001, 10.0, 1.1, -10.0, 5.0, 0.1], [2., 20.0, 1.99, 1.99, 50.0, 4.99, 2., 20.0, 1.99, 1.99, 50.0, 4.99]]

    # Generate simulation parameter points. Priors hard-coded for now.
    prior = utils.BoxUniform(low=torch.tensor([0.001] + prior_poiss[0] + prior_ps[0]), high=torch.tensor([2.] + prior_poiss[1] + prior_ps[1]))
    thetas = prior.sample((n,))
    results["theta"] = thetas

    # Generate maps
    x = [simulator(theta.detach().numpy(), temps_poiss, temps_ps, mask_sim, mask_roi, kp.psf_fermi_r) for theta in tqdm(thetas)]
    results["x"] = x

    return results


def save(data_dir, name, data):
    """ Save simulated data to file
    """

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
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Main high-level script that starts the GCE simulations")

    parser.add_argument(
        "-n", type=int, default=10000, help="Number of samples to generate. Default is 10k.",
    )
    parser.add_argument("--dif", type=str, default="ModelO", help='Diffuse model to simulate, wither "ModelO" (default) or "p6"')
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
    results = simulate(n=args.n, dif=args.dif)
    save(args.dir, name, results)

    logger.info("All done! Have a nice day!")
