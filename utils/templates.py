import sys
sys.path.append("../")

import numpy as np
import healpy as hp

from utils import create_mask as cm

def mod(dividends, divisor):
    """ Return dividends (array) mod divisor (double)
        Stolen from Nick's code
    """

    output = np.zeros(len(dividends))

    for i in range(len(dividends)): 
        output[i] = dividends[i]
        done=False
        while (not done):
            if output[i] >= divisor:
                output[i] -= divisor
            elif output[i] < 0.:
                output[i] += divisor
            else:
                done=True

    return output


def rho_NFW(r, gamma=1., r_s=20.):
    """ Generalized NFW profile
    """
    return (r / r_s) ** -gamma * (1 + (r / r_s)) ** (-3 + gamma) 

def rGC(s_ary, b_ary, l_ary, rsun=8.34):
    """ Distance to GC as a function of LOS distance, latitude, longitude
    """
    return np.sqrt(s_ary ** 2 - 2. * rsun * np.transpose(np.multiply.outer(s_ary, np.cos(b_ary) * np.cos(l_ary))) + rsun ** 2)

def get_NFW2_template(gamma=1.2, nside=128):


    mask = cm.make_mask_total(nside=nside, band_mask = True, band_mask_range = 0,
                                mask_ring = True, inner = 0, outer = 50)
    mask_restrict = np.where(mask == 0)[0]

    # Get lon/lat array

    theta_ary, phi_ary = hp.pix2ang(nside, mask_restrict)

    b_ary = np.pi / 2. - theta_ary
    l_ary = mod(phi_ary + np.pi, 2. * np.pi) - np.pi

    s_ary = np.linspace(0, 30, 200)

    # LOS integral of density^2
    int_rho2_temp = np.trapz(rho_NFW(rGC(s_ary, b_ary, l_ary), gamma=gamma) ** 2, s_ary, axis=1)
    int_rho2_temp /= np.mean(int_rho2_temp)

    int_rho2 = np.zeros(hp.nside2npix(128))
    int_rho2[~mask] = int_rho2_temp

    return int_rho2
