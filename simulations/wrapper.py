import numpy as np

from simulations.simulate_ps import SimulateMap
from models.scd import dnds


def simulator(theta, mask, temp_ps, psf_r_func):
    the_map = np.zeros(np.sum(~mask))
    while (np.sum(the_map) == 0) or np.sum(np.isnan(the_map)) or np.sum(np.isinf(the_map)):
        s_ary = np.logspace(-2, 2, 100)
        theta[0] = 10 ** theta[0]
        dnds_ary = dnds(s_ary, theta)
        sm = SimulateMap([], [], [s_ary], [dnds_ary], [temp_ps], psf_r_func, n_exp=[theta[0]])
        the_map = sm.create_map()
        the_map = the_map[~mask].astype(np.float32)
        the_map /= np.mean(the_map)
        the_map = the_map.reshape((1, -1))
    return the_map
