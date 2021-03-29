import argparse
import logging

import numpy as np
import healpy as hp
from tqdm import tqdm

from utils.utils import ring2nest
from utils import create_mask as cm

def parse_args():
    parser = argparse.ArgumentParser(description="High-level script for the training of the neural likelihood ratio estimators")

    # Main options
    parser.add_argument("--sample", type=str, default="train_ModelO_gamma_fix", help='Sample name, like "train"')
    parser.add_argument("--i_start", type=int, default=0, help='Index at which to start')
    parser.add_argument("--n_files", type=int, default=10, help='How many files to loop through')

    # Training option
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO,
    )
    logging.info("Hi!")

    args = parse_args()

    hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask = True, band_mask_range = 0, mask_ring = True, inner = 0, outer = 25)
    mask = hp.ud_grade(hp_mask_nside1, 128)

    sample = args.sample 

    for i in tqdm(range(args.i_start, args.i_start + args.n_files)):

        filename = "../data/samples/x_{}_{}.npy".format(sample, i)
        x_og = np.load(filename)
        x_og = ring2nest(x_og.squeeze(), mask)
        x_og = np.expand_dims(x_og, 1)
        filename_save = "../data/samples/x_{}_nest_{}.npy".format(sample, i)
        np.save(filename_save, x_og)

    logging.info("All done! Have a nice day!")
