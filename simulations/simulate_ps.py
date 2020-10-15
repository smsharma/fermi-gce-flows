import sys

sys.path.append("../")

import numpy as np
import healpy as hp
from tqdm.autonotebook import tqdm

from utils.pdf_sampler import PDFSampler


class DrawSources:
    def __init__(self, S_ary=None, dNdS_ary=None, n_exp=None):
        """ Draw sources following an SCD and create a map. Based on NPTFit-Sim 
            (https://github.com/nickrodd/NPTFit-Sim).
            :param S_ary: Array of counts, spaced linearly in log-space
            :param dNdS_ary: Corresponding SCD dN/dS array (full sky)
            :param n_exp: Number of sources to draw. If not provided, obtained by integrating the SCD. 
        """

        self.S_ary = S_ary
        self.dNdS_ary = dNdS_ary
        self.n_exp = n_exp

        if self.S_ary is not None:
            self.init_ps()

    def init_ps(self):

        # If not provided, obtain expected number of sources by integrating SCD
        if self.n_exp is None:
            self.n_exp = np.trapz(self.dNdS_ary, self.S_ary)

        self.n_draw = np.random.poisson(self.n_exp)

        # Get dS array, assuming counts array is space linearly in log-space

        logS_ary = np.log10(self.S_ary)
        dlogS_ary = np.diff(logS_ary)[0]  # Spacing in log-space

        S_for_dS_ary = np.logspace(logS_ary[0] - dlogS_ary / 2.0, logS_ary[-1] + dlogS_ary / 2.0, len(self.S_ary) + 1)
        dS_ary = np.diff(S_for_dS_ary)

        # Sample, accounting for dS factor for log-space sampling
        self.sample = PDFSampler(self.S_ary, dS_ary * self.dNdS_ary)

        # Expected and realized counts
        self.counts_expected_sample = self.sample(self.n_draw)
        self.counts_sample = np.random.poisson(self.counts_expected_sample)

    def get_coords(self, temp, n_ps, n_sample=1000):
        """ Get PS coordinates for a given template using rejection sampling. 
            Based on NPTFit-Sim (https://github.com/nickrodd/NPTFit-Sim).
            :param temp: The PS template
            :param n_ps: Number of PSs to generate
            :param n_sample: Number of samples to take at a given time. Set to a sensible default.
            :return: (theta, phi) coordinates of PSs following template
        """

        n_accept = 0
        th_ary_accept = []
        ph_ary_accept = []

        # Do rejection sampling until more than the required number of PS coordinates are produced
        while n_accept < n_ps:

            th_ary = 2 * np.arcsin(np.sqrt(np.random.random(n_sample)))
            ph_ary = 2 * np.pi * np.random.random(n_sample)

            temp /= np.max(temp)
            nside = hp.npix2nside(len(temp))

            rnd = np.random.random(n_sample)
            pix = hp.ang2pix(nside, th_ary, ph_ary)

            accept = rnd <= temp[pix]

            th_ary_accept += list(th_ary[accept])
            ph_ary_accept += list(ph_ary[accept])

            n_accept += np.sum(accept)

        return th_ary_accept[:n_ps], ph_ary_accept[:n_ps]

    def create_ps_map(self, temp, psf_r=None):
        """ Add PS photon counts to map following a given PSF description and template.
            Based on NPTFit-Sim (https://github.com/nickrodd/NPTFit-Sim).
            :param temp: Template for spatial distribution of PSs
            :param psf_r: Radial PSF
        """
        # Sample the radial PSF to later determine placement of photons.
        f = np.linspace(0.0, np.pi, 1000000)
        pdf_psf = f * psf_r(f)
        pdf = PDFSampler(f, pdf_psf)
        nside = hp.npix2nside(len(temp))

        # Draw coordinates according to template
        self.th_ary, self.ph_ary = self.get_coords(temp, self.n_draw)

        the_map = np.zeros(hp.nside2npix(nside))

        for ips in tqdm(range(self.n_draw)):
            th = self.th_ary[ips]
            ph = self.ph_ary[ips]
            num_phot = self.counts_sample[ips]
            phm = ph + np.pi / 2.0
            rotx = np.matrix([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
            rotz = np.matrix([[np.cos(phm), -np.sin(phm), 0], [np.sin(phm), np.cos(phm), 0], [0, 0, 1]])
            dist = pdf(num_phot)
            randPhi = 2 * np.pi * np.random.random(num_phot)
            X = hp.ang2vec(dist, randPhi).T
            Xp = rotz * (rotx * X)
            posit = np.array(hp.vec2pix(nside, *Xp))
            np.add.at(the_map, posit, 1)

        return the_map


class SimulateMap:
    def __init__(self, temps, norms, S_arys=None, dNdS_arys=None, ps_temps=None, psf_r=None, nside=128):

        self.temps = temps
        self.norms = norms
        self.nside = nside
        self.S_arys = S_arys
        self.dNdS_arys = dNdS_arys
        self.ps_temps = ps_temps
        self.psf_r = psf_r

        # ud_grade templates
        for i_temp in range(len(self.temps)):
            self.temps[i_temp] = hp.ud_grade(self.temps[i_temp], nside_out=self.nside, power=-2)

        for i_temp in range(len(self.ps_temps)):
            self.ps_temps[i_temp] = hp.ud_grade(self.ps_temps[i_temp], nside_out=self.nside, power=-2)

    def create_map(self):
        self.mu_map = np.zeros(hp.nside2npix(self.nside))

        for i_temp in range(len(self.temps)):
            self.mu_map += self.norms[i_temp] * self.temps[i_temp]

        gamma_map = np.random.poisson(self.mu_map)

        if self.S_arys is not None:
            for i_ps in range(len(self.ps_temps)):
                ds = DrawSources(self.S_arys[i_ps], self.dNdS_arys[i_ps])
                self.ps_map = ds.create_ps_map(self.ps_temps[i_ps], self.psf_r)
                gamma_map += self.ps_map.astype(np.int64)

        return gamma_map.astype(np.int32)

