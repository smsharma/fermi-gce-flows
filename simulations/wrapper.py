import numpy as np

from simulations.simulate_ps import SimulateMap
from models.scd import dnds


def simulator(theta, temps_poiss, temps_ps, mask_sim, mask_roi, psf_r_func, norm=False):

    the_map = np.zeros(np.sum(~mask_sim) + 2)
    s_ary = np.logspace(-1, 2, 100)

    good_map = False

    while not good_map:
        


        # Normalize poiss DM norm to get correct counts/pix in ROI
        norm_gce = theta[0] / np.mean(temps_poiss[0][np.where(~mask_sim)])

        norms_poiss =  theta[1:len(temps_poiss)]

        # Normalize PS map to get correct counts/pix in ROI
        dnds_ary = []
        idx_theta_ps = len(temps_poiss)
        for temp_ps in temps_ps:
            dnds_ary_temp = dnds(s_ary, theta[idx_theta_ps:idx_theta_ps + 6])
            s_exp = np.trapz(s_ary * dnds_ary_temp, s_ary)
            temp_ratio = np.sum(temp_ps[np.where(~mask_sim)]) / np.sum(temp_ps)
            dnds_ary_temp *= theta[idx_theta_ps] * np.sum(~mask_sim) / s_exp / temp_ratio
            dnds_ary.append(dnds_ary_temp)
            idx_theta_ps += 6

        sm = SimulateMap(temps_poiss, [norm_gce] + list(norms_poiss), [s_ary] * len(temps_ps), dnds_ary, temps_ps, psf_r_func)
        the_map_temp = sm.create_map()
        the_map_temp[mask_roi] = 0.
        the_map[:-2] = the_map_temp[~mask_sim].astype(np.float32)

        mean_map = np.mean(the_map)
        var_map = np.var(the_map)

        if norm:
            the_map[:-1] /= mean_map

        # Append auxiliary variables to data array
        the_map[-2] = np.log(mean_map)
        the_map[-1] = np.log(np.sqrt(var_map))

        the_map = the_map.reshape((1, -1))

        # Resimulate if map is crap
        if (np.sum(the_map) == 0) or np.sum(np.isnan(the_map)) or np.sum(np.isinf(the_map)):
            good_map = False
        else:
            good_map = True

    return the_map
