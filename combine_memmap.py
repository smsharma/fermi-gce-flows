#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import numpy as np
import re
from tqdm import *

# combined = np.memmap('data/samples/x_train_ModelO_gamma_default_1M_memmap.npy', dtype='float32', mode='w+', shape=(1000000, 1, 16384))

# del combined

for i in tqdm(range(100)):
    b = np.memmap('data/samples/x_train_ModelO_gamma_default_1M_memmap.npy', dtype='float32', mode='r+', shape=(1000, 1, 16384), offset=int(1000 * i * 1 * 16384 * 32 / 8))
    a = np.load('data/samples/x_train_ModelO_gamma_default_{}.npy'.format(i))
    # a = np.memmap('data/samples/x_train_ModelO_gamma_default_{}.npy'.format(i), dtype='float32', mode='c', shape=(1000, 1, 16384))
    b[:] = a[:]
    del a, b
