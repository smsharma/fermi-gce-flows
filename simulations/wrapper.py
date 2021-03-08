import logging
logger = logging.getLogger(__name__)

import numpy as np

from simulations.simulate_ps import SimulateMap
from models.scd import dnds


def simulator(theta, temps_poiss, temps_ps, mask_sim, mask_normalize_counts, mask_roi, psf_r_func):

    the_map = np.zeros(np.sum(~mask_sim))
    aux_vars = np.zeros(2)
    s_ary = np.logspace(-1, 2, 100)

    good_map = False

    while not good_map:
        
        # Normalize poiss DM norm to get correct counts/pix in ROI
        norm_gce = theta[0] / np.mean(temps_poiss[0][np.where(~mask_sim)])

        # Grab the rest of the poiss norms
        norms_poiss =  theta[1:len(temps_poiss)]
        
        # Normalize PS map to get correct counts/pix in ROI
        # and construct appropriate dnds arrays for each PS template

        dnds_ary = []
        idx_theta_ps = len(temps_poiss)
        for temp_ps in temps_ps:
            dnds_ary_temp = dnds(s_ary, theta[idx_theta_ps:idx_theta_ps + 6])
            s_exp = np.trapz(s_ary * dnds_ary_temp, s_ary)
            temp_ratio = np.sum(temp_ps[np.where(~mask_normalize_counts)]) / np.sum(temp_ps)
            dnds_ary_temp *= theta[idx_theta_ps] * np.sum(~mask_normalize_counts) / s_exp / temp_ratio
            dnds_ary.append(dnds_ary_temp)
            idx_theta_ps += 6

        # Draw PSs and simulate map
        sm = SimulateMap(temps_poiss, [norm_gce] + list(norms_poiss), [s_ary] * len(temps_ps), dnds_ary, temps_ps, psf_r_func)
        the_map_temp = sm.create_map()
        the_map_temp[mask_roi] = 0.
        the_map = the_map_temp[~mask_sim].astype(np.float32)

        # Grab auxiliary variables
        mean_map = np.mean(the_map)
        var_map = np.var(the_map)

        the_map = the_map.reshape((1, -1))
        aux_vars = np.array([np.log(mean_map), np.log(np.sqrt(var_map))]).reshape((1, -1))

        # Resimulate if map is crap
        if (np.sum(the_map) == 0) or np.sum(np.isnan(the_map)) or np.sum(np.isinf(the_map)):
            good_map = False
            logger.info("Resimulating a crap map...")
        else:
            good_map = True

    return (the_map, aux_vars)
