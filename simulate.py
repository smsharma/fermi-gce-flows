from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
import argparse
import logging
from operator import itemgetter

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
from utils.templates import get_NFW2_template
from models.psf import KingPSF


def simulate(n=10000, r_outer=25, nside=128, psf="king", dif="ModelO", gamma="fix"):
    """ High-level simulation script
    """

    # Get mask of central pixel for nside=1
    hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask=True, band_mask_range=0, mask_ring=True, inner=0, outer=r_outer)

    # Get mask corresponding to nside=128
    mask_sim = hp.ud_grade(hp_mask_nside1, nside)

    # ROi to normalize counts over
    mask_normalize_counts = cm.make_mask_total(nside=nside, band_mask = True, band_mask_range=2, mask_ring=True, inner=0, outer=25.)

    # Get ROI mask
    ps_mask = np.load("data/mask_3fgl_0p8deg.npy")
    mask_roi = cm.make_mask_total(nside=nside, band_mask = True, band_mask_range=2, mask_ring=True, inner=0, outer=r_outer, custom_mask=ps_mask)

    # King PSF hard-coded for now
    if psf == "king":
        kp = KingPSF()
    else:
        raise NotImplementedError

    # Load standard templates
    temp_gce = np.load("data/fermi_data/template_gce.npy")
    temp_dif = np.load("data/fermi_data/template_dif.npy")
    temp_psc = np.load("data/fermi_data/template_psc.npy")
    temp_iso = np.load("data/fermi_data/template_iso.npy")
    temp_dsk = np.load("data/fermi_data/template_dsk.npy")
    temp_bub = np.load("data/fermi_data/template_bub.npy")

    # Load Model O templates
    temp_mO_pibrem = np.load('data/fermi_data/ModelO_r25_q1_pibrem.npy')
    temp_mO_ics = np.load('data/fermi_data/ModelO_r25_q1_ics.npy')

    logger.info("Generating training data with %s maps", n)

    # Dict to save results
    results = {}

    # Priors for DM template, if required
    
    if gamma == "fix":
        prior_temp = [[],[]]
    elif gamma == "float":
        prior_temp = [[0.5],[1.5]]
    elif gamma == "float_both":
        prior_temp = [[0.5,0.5],[1.5,1.5]]
    else:
        raise NotImplementedError

    # gce, dsk PS priors
    prior_ps = [[0.001, 10.0, 1.1, -10.0, 5.0, 0.1, 0.001, 10.0, 1.1, -10.0, 5.0, 0.1], [2.5, 20.0, 1.99, 1.99, 50.0, 4.99, 2.5, 20.0, 1.99, 1.99, 50.0, 4.99]]

    # Poiss priors

    if dif == "ModelO":
        # iso, bub, psc, dif_pibrem, dif_ics
        prior_poiss = [[0.001, 0.001, 0.001, 6., 1.], [1.5, 1.5, 1.5, 12., 6.]]
    elif dif == "p6v11":
        # iso, bub, psc, dif
        prior_poiss = [[0.001, 0.001, 0.001, 11.], [1.5, 1.5, 1.5, 16.]]
    else:
        raise NotImplementedError

    # Generate simulation parameter points. Priors hard-coded for now.
    prior = utils.BoxUniform(low=torch.tensor([prior_ps[0][0]] + prior_poiss[0] + prior_ps[0] + prior_temp[0]), 
                             high=torch.tensor([prior_ps[1][0]] + prior_poiss[1] + prior_ps[1] + prior_temp[1]))

    thetas = prior.sample((n,))

    # Generate NFW template

    logger.info("Generating NFW template...")

    if gamma == "fix":
        temp_gce = get_NFW2_template(gamma=1.2)
        temps_gce_poiss = [temp_gce] * n
        temps_gce_ps = [temp_gce] * n
    elif gamma == "float":
        temps_gce = [get_NFW2_template(gamma=gamma.detach().numpy()) for gamma in tqdm(thetas[:, -1])]
        temps_gce_poiss = temps_gce
        temps_gce_ps = temps_gce
    elif gamma == "float_both":
        temps_gce_poiss = [get_NFW2_template(gamma=gamma.detach().numpy()) for gamma in tqdm(thetas[:, -2])]
        temps_gce_ps = [get_NFW2_template(gamma=gamma.detach().numpy()) for gamma in tqdm(thetas[:, -1])]
    else:
        raise NotImplementedError

    # List of templates except GCE template

    temps_ps = [temp_dsk]

    if dif == "ModelO":
        temps_poiss = [temp_iso, temp_bub, temp_psc, temp_mO_pibrem, temp_mO_ics]
    elif dif == "p6v11":
        temps_poiss = [temp_iso, temp_bub, temp_psc, temp_dif]
    else:
        raise NotImplementedError

    # Generate maps

    logger.info("Generating maps...")
    
    x_and_aux = [simulator(theta.detach().numpy(), [temp_gce_poiss] + temps_poiss, [temp_gce_ps] + temps_ps, mask_sim, mask_normalize_counts, mask_roi, kp.psf_fermi_r) for (theta, temp_gce_poiss, temp_gce_ps) in tqdm(zip(thetas, temps_gce_poiss, temps_gce_ps))]

    # Grab maps and aux variables
    x = torch.Tensor(list(map(itemgetter(0), x_and_aux)))
    x_aux = torch.Tensor(list(map(itemgetter(1), x_and_aux)))

    results["x"] = x
    results["x_aux"] = x_aux
    results["theta"] = thetas

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
    parser.add_argument("--gamma", type=str, default="fix", help='Whether to float NFW index gamma. "fix" (default, fixes to gamma=1.2), "float" (float both gammas), or "float_both" (float PS and poiss gammas separately)')
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
    results = simulate(n=args.n, dif=args.dif, gamma=args.gamma)
    save(args.dir, name, results)

    logger.info("All done! Have a nice day!")
