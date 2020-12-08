import os
import numpy as np


def load_and_check(filename, use_memmap=False):
    # Don't load image files > 1 GB into memory
    if use_memmap and os.stat(filename).st_size > 1.0 * 1024 ** 3:
        data = np.load(filename, mmap_mode="c")
    else:
        data = np.load(filename)
    return data
