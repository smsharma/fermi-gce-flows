import numpy as np

from simulations.simulate_ps import SimulateMap
from models.scd import dnds


def simulator(theta, mask, temp_ps, psf_r_func):

    the_map = np.zeros(np.sum(~mask) + 1)
    mean_map = 0
    good_map = False

    while not good_map:
        s_ary = np.logspace(-1, 2, 100)
        dnds_ary = dnds(s_ary, theta[1:])
        temp_ratio = np.sum(temp_ps[np.where(~mask)]) / np.sum(temp_ps)
        norm_poiss = theta[0] / np.mean(temp_ps[np.where(~mask)])
        s_exp = np.trapz(s_ary * dnds_ary, s_ary)
        dnds_ary *= theta[1] * np.sum(~mask) / s_exp / temp_ratio
        sm = SimulateMap([temp_ps], [norm_poiss], [s_ary], [dnds_ary], [temp_ps], psf_r_func)
        the_map_temp = sm.create_map()
        the_map[:-1] = the_map_temp[~mask].astype(np.float32)
        mean_map = np.mean(the_map)
        the_map[:-1] /= mean_map
        the_map[-1] = mean_map
        the_map = the_map.reshape((1, -1))

        if (np.sum(the_map) == 0) or np.sum(np.isnan(the_map)) or np.sum(np.isinf(the_map)):
            good_map = False
        else:
            good_map = True

    return the_map
