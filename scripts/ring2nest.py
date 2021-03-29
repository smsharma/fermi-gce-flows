import sys
sys.path.append("../")

import numpy as np
import healpy as hp
from tqdm import tqdm

from utils.utils import ring2nest
from utils import create_mask as cm

hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask = True, band_mask_range = 0, mask_ring = True, inner = 0, outer = 25)
mask = hp.ud_grade(hp_mask_nside1, 128)

sample = "train_ModelO_gamma_fix"
n_files = 1200

for i in tqdm(range(n_files)):

    filename = "../data/samples/x_{}_{}.npy".format(sample, i)
    x_og = np.load(filename)
    x_og = ring2nest(x_og.squeeze(), mask)
    x_og = np.expand_dims(x_og, 1)
    filename_save = "../data/samples/x_{}_nest_{}.npy".format(sample, i)
    np.save(filename_save, x_og)
