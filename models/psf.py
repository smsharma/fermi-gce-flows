import numpy as np


class KingPSF:
    def __init__(self):

        # Define parameters that specify the Fermi-LAT PSF at 2 GeV
        self.fcore = 0.748988248179
        self.score = 0.428653790656
        self.gcore = 7.82363229341
        self.stail = 0.715962650769
        self.gtail = 3.61883748683
        self.spe = 0.00456544262478

    def king_fn(self, x, sigma, gamma):
        """ Define the full PSF in terms of two King functions
        """
        return 1.0 / (2.0 * np.pi * sigma ** 2.0) * (1.0 - 1.0 / gamma) * (1.0 + (x ** 2.0 / (2.0 * gamma * sigma ** 2.0))) ** (-gamma)

    def psf_fermi_r(self, r):
        return self.fcore * self.king_fn(r / self.spe, self.score, self.gcore) + (1 - self.fcore) * self.king_fn(r / self.spe, self.stail, self.gtail)

    def psf_gauss_r(self, r, psf_sigma_deg):

        # Define parameters that specify the PSF
        sigma = psf_sigma_deg * np.pi / 180.0

        # Lambda function to pass user defined PSF
        return np.exp(-(r ** 2.0) / (2.0 * sigma ** 2.0))
