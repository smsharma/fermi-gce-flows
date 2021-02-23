import pymaster as nmt
from tqdm import tqdm
import numpy as np
import healpy as hp


def construct_histogram(X, bins_lower=0, bins_upper=96, axis=2):
    """ Construct histogram summary statistics. X should be of shape (n_datasets, 1, n_features).
    """
    X_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=np.arange(bins_lower,bins_upper))[0], axis, X)
    return X_hist

def construct_power_spectrum(X, mask_map, mask_roi, ells_per_bandpower=4, nside=128):
    """ Construct histogram summary statistics. X should be of shape (n_datasets, 1, n_features).
    """

    n_data = X.shape[0]

    # Construct full-sky embeddings
    test_maps = np.zeros((n_data, hp.nside2npix(nside)))
    test_maps[:, np.where(~mask_map)[0]] = X[:,0,:]

    # Initialize binning scheme
    b = nmt.NmtBin.from_nside_linear(nside, ells_per_bandpower)

    cl_00_list = []
    for i_maps in tqdm(range(n_data)):
        f_0 = nmt.NmtField(~mask_roi, [test_maps[i_maps]])
        cl_00 = nmt.compute_full_master(f_0, f_0, b)
        cl_00_list.append(cl_00)

    cl_00_ary = np.array(cl_00_list)

    return cl_00_ary
    