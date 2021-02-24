from pathlib import Path
import argparse
import joblib
import logging
import json

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm
import healpy as hp
from utils import create_mask as cm
from simulations.summaries import construct_histogram, construct_power_spectrum

logger = logging.getLogger(__name__)


def do_pca(sample, n_files, n_components):

    # Incremental PCA fit

    incremental_pca = IncrementalPCA(n_components=n_components)

    for i_file in tqdm(range(*n_files)):

        filename = "data/samples/x_{}_{}.npy".format(sample, i_file)
        if not Path(filename).is_file():
            continue
        X = np.load(filename)
        
        incremental_pca.partial_fit(X[:,0,:])

    # Save PCA model

    model_filename = "data/models/pca_{}_{}.p".format(n_components, sample)

    joblib.dump(incremental_pca, model_filename)
    incremental_pca = joblib.load(model_filename)  # Load model to make sure that it works

    # Do PCA decomposition

    for i_file in tqdm(range(*n_files)):

        filename = "data/samples/x_{}_{}.npy".format(sample, i_file)
        if not Path(filename).is_file():
            continue
        X = np.load(filename)
        X_pca = incremental_pca.transform(X[:,0,:])
        X_pca = np.expand_dims(X_pca, 1)  # Add channel dimension
        np.save("data/samples/x_pca_{}_{}_{}.npy".format(n_components, sample, i_file), X_pca)

def do_histogram(sample, n_files, n_max_bins):

    # Construct counts histograms

    for i_file in tqdm(range(*n_files)):

        filename = "data/samples/x_{}_{}.npy".format(sample, i_file)
        if not Path(filename).is_file():
            continue
        X = np.load(filename)
        X_hist = construct_histogram(X)
        
        np.save("data/samples/x_hist_{}_{}_{}.npy".format(n_max_bins, sample, i_file), X_hist)

def do_power_spectrum(sample, n_files, ells_per_bandpower):

    nside = 128

    # Get mask corresponding to nside=1
    hp_mask_nside1 = cm.make_mask_total(nside=1, mask_ring=True, inner=0, outer=25)

    # Get mask corresponding to nside=128
    mask_sim = hp.ud_grade(hp_mask_nside1, nside)

    # Get ROI mask
    ps_mask = np.load("data/mask_3fgl_0p8deg.npy")
    mask_roi = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=25, custom_mask=ps_mask)

    # Construct power spectrum

    for i_file in tqdm(range(*n_files)):

        filename = "data/samples/x_{}_{}.npy".format(sample, i_file)
        if not Path(filename).is_file():
            continue
        X = np.load(filename)
        X_pspec = construct_power_spectrum(X, mask_sim, mask_roi, ells_per_bandpower)

        np.save("data/samples/x_pspec_{}_{}_{}.npy".format(ells_per_bandpower, sample, i_file), X_pspec)

def parse_args():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="High-level script that extract low-dim summaries")

    parser.add_argument('--do_pca', dest='do_pca', action='store_true')
    parser.add_argument('--do_histogram', dest='do_histogram', action='store_true')
    parser.add_argument('--do_power_spectrum', dest='do_power_spectrum', action='store_true')
    parser.add_argument("--debug", action="store_true", help="Prints debug output.")

    parser.add_argument("--sample", type=str, default=None, help='Sample name, like "train" or "test".')
    parser.add_argument("--n_components", type=int, default=96, help="Number of PCA dimensions",)
    parser.add_argument("--n_max_bins", type=int, default=96, help="Maximum bin edge of histogram",)
    parser.add_argument("--ells_per_bandpower", type=int, default=4, help="Multipole binning in power spectrum",)
    parser.add_argument("--n_files", type=str, default="[0,500]", help='Which files to act on')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO,)

    logger.info("Hi!")

    args.n_files = list(json.loads(args.n_files))

    if args.do_pca:
        logger.info("Doing PCA decomposition")
        do_pca(args.sample, args.n_files, args.n_components)
    
    if args.do_histogram:
        logger.info("Constructing histograms")
        do_histogram(args.sample, args.n_files, args.n_max_bins)

    if args.do_power_spectrum:
        logger.info("Doing power spectrum decomposition")
        do_power_spectrum(args.sample, args.n_files, args.ells_per_bandpower)

    logger.info("All done! Have a nice day!")
