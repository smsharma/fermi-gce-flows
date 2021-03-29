import os
import numpy as np
import healpy as hp

def load_and_check(filename, use_memmap=False):
    # Don't load image files > 1 GB into memory
    if use_memmap and os.stat(filename).st_size > 1.0 * 1024 ** 3:
        data = np.load(filename, mmap_mode="c")
    else:
        data = np.load(filename)
    return data

def ring2nest(the_map, the_mask, nside=128, return_masked=True):
    embed_map = np.zeros((the_map.shape[0], hp.nside2npix(nside)))
    embed_map[:, ~the_mask] = the_map
    the_map = hp.reorder(embed_map, r2n=True)
    if return_masked:
        the_mask = hp.reorder(the_mask, r2n=True)
        return the_map[:, ~the_mask]
    return the_map