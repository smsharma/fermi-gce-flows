import os

import numpy as np
import torch


def fibonacci_sphere(samples=1):
    """ Uniform sampling on a sphere
    """

    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points


def intersect1d(tensor1, tensor2):
    """ (1-D) Intersection of two tensors
    """
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def make_dirs(dirs):
    """ Creates directories if they do not already exist 
    """

    for d in dirs:
        if not os.path.exists(d):
            try:
                os.mkdir(d)
            except OSError as e:
                if e.errno != 17:
                    raise   
