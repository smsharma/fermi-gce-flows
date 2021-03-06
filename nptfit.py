import argparse
import numpy as np
import healpy as hp
import torch
from tqdm import *
import dynesty
from multiprocessing import Pool
from dynesty import DynamicNestedSampler, NestedSampler
from dynesty.utils import resample_equal

from models.scd import dnds
from models.likelihoods import log_like_np
from models.nptf_model import NPRegression
from utils.templates import get_NFW2_template
from utils.psf_correction import PSFCorrection
from utils import create_mask as cm
from models.psf import KingPSF

from NPTFit import npll


class NPRegression:
    def __init__(self, temps_poiss, temps_ps, priors_poiss, data, priors_ps=None, f_ary=[1.], df_rho_div_f_ary=[1.], roi_mask=None, param_log=None, transform_prior_on_s=True):
        
        self.transform_prior_on_s = transform_prior_on_s
        
        self.temps_poiss= temps_poiss
        self.temps_ps = temps_ps
        self.priors_poiss = priors_poiss
        self.priors_ps = priors_ps
        self.data = data.astype(np.int32)
        self.roi_mask = roi_mask
        
        if self.priors_ps is not None:
            self.priors = np.concatenate([np.transpose(self.priors_poiss), np.transpose(self.priors_ps)])
        else:
            self.priors = np.transpose(self.priors_poiss)
            
        self.priors_lo = self.priors[:, 0]
        self.priors_interval = self.priors[:, 1] - self.priors[:, 0]
 
        self.f_ary = f_ary
        self.df_rho_div_f_ary = df_rho_div_f_ary
        
        self.n_poiss = len(temps_poiss)
        self.n_ps = len(temps_ps)
        self.n_ps_params = 6
        self.n_params = self.n_poiss + self.n_ps_params * self.n_ps
        
        self.param_log = param_log

        if self.roi_mask is not None:
            self.temps_poiss = np.array([temp[~self.roi_mask] for temp in self.temps_poiss])
            self.temps_ps = np.array([temp[~self.roi_mask] for temp in self.temps_ps])
            self.data = np.array(self.data[~self.roi_mask])
                
    def loglike(self, theta):
        
        theta[self.param_log] = 10 ** theta[self.param_log]
        theta_poiss = theta[:self.n_poiss]

        theta_ps = np.array(np.split(theta[self.n_poiss:], self.n_ps))
            
        if self.transform_prior_on_s:

            theta_poiss[0] /= np.mean(self.temps_poiss[0])

            for i_ps in torch.arange(self.n_ps):

                s_ary = torch.logspace(-2, 2, 100)
                s_exp_temp = theta_ps[i_ps][0]
                theta_ps[i_ps][0] = 1.
                dnds_ary_temp = dnds(s_ary, theta_ps[i_ps])
                s_exp = np.mean(self.temps_ps[i_ps]) * np.trapz(s_ary * dnds_ary_temp, s_ary)

                theta_ps[i_ps][0] = s_exp_temp / s_exp
        
        pt_sum_compressed = np.sum(self.temps_poiss * theta_poiss[:, np.newaxis], axis=0)
        
        return npll.log_like(pt_sum_compressed, theta_ps, self.f_ary, self.df_rho_div_f_ary, self.temps_ps, self.data)

    def prior_cube(self, u, ndim=1, nparams=1):

        u *= self.priors_interval 
        u += self.priors_lo
        
        return u

    def run_dynesty(self, nlive=200, n_cpus=4):

        n_dim = self.n_params

        with Pool(processes=n_cpus) as pool:

            sampler = NestedSampler(self.loglike, self.prior_cube, n_dim, pool=pool, queue_size=n_cpus, nlive=nlive)
            sampler.run_nested(dlogz=1.)

        # Draw posterior samples
        weights = np.exp(sampler.results['logwt'] - sampler.results['logz'][-1])
        samples_weighted = resample_equal(sampler.results.samples, weights)

        return sampler.results.samples, samples_weighted

def load_psf():
    # # Define parameters that specify the Fermi-LAT PSF at 2 GeV
    # fcore = 0.748988248179
    # score = 0.428653790656
    # gcore = 7.82363229341
    # stail = 0.715962650769
    # gtail = 3.61883748683
    # spe = 0.00456544262478

    # # Define the full PSF in terms of two King functions
    # def king_fn(x, sigma, gamma):
    #     return 1./(2.*np.pi*sigma**2.)*(1.-1./gamma)*(1.+(x**2./(2.*gamma*sigma**2.)))**(-gamma)

    # def Fermi_PSF(r):
    #     return fcore*king_fn(r/spe,score,gcore) + (1-fcore)*king_fn(r/spe,stail,gtail)

    # # Modify the relevant parameters in pc_inst and then make or load the PSF
    # pc_inst = PSFCorrection(delay_compute=True)
    # pc_inst.psf_r_func = lambda r: Fermi_PSF(r)
    # pc_inst.sample_psf_max = 10.*spe*(score+stail)/2.
    # pc_inst.psf_samples = 10000
    # pc_inst.psf_tag = 'Fermi_PSF_2GeV'
    # pc_inst.make_or_load_psf_corr()

    # # Extract f_ary and df_rho_div_f_ary as usual
    # f_ary = pc_inst.f_ary
    # df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

    pc_inst = PSFCorrection(psf_sigma_deg=0.1812)

    return pc_inst.f_ary, pc_inst.df_rho_div_f_ary

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_cpus", action="store", default=36, type=int)
    parser.add_argument("--n_live", action="store", default=500, type=int)
    parser.add_argument("--sample_name", action="store", default="runs", type=str)
    parser.add_argument("--sampler", action="store", default="dynesty", type=str)
    parser.add_argument("--save_dir", action="store", default="data/nptfit_samples/", type=str)
    parser.add_argument("--r_outer", action="store", default=30., type=float)
    parser.add_argument("--ps_mask_type", action="store", default="0.8", type=str)
    parser.add_argument("--transform_prior_on_s", action="store", default=0, type=int)
    parser.add_argument("--diffuse", action="store", default="ModelO", type=str)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Load templates
    temp_gce = np.load("data/fermi_data/template_gce.npy") # get_NFW2_template(gamma=1.2)
    temp_dif = np.load("data/fermi_data/template_dif.npy")
    temp_psc = np.load("data/fermi_data/template_psc.npy")
    temp_iso = np.load("data/fermi_data/template_iso.npy")
    temp_dsk = np.load("data/fermi_data/template_dsk.npy")
    temp_bub = np.load("data/fermi_data/template_bub.npy")

    temp_mO_pibrem = np.load('data/fermi_data/ModelO_r25_q1_pibrem.npy')
    temp_mO_ics = np.load('data/fermi_data/ModelO_r25_q1_ics.npy')

    temp_mA_pibrem = hp.ud_grade(np.load('data/modelA/modelA_brempi0.npy'), nside_out=128, power=-2)
    temp_mA_ics = hp.ud_grade(np.load('data/modelA/modelA_ics.npy'), nside_out=128, power=-2)

    fermi_exp = np.load("data/fermi_data/fermidata_exposure.npy")
    fermi_data = np.load("data/fermi_data/fermidata_counts.npy")

    if args.ps_mask_type == "0.8":
        ps_mask = np.load("data/mask_3fgl_0p8deg.npy")
    elif args.ps_mask_type == "default":
        ps_mask = np.load("data/fermi_data/fermidata_pscmask.npy") == 1
    else:
        raise NotImplementedError

    roi_mask = cm.make_mask_total(nside=128, band_mask = True, band_mask_range=2, mask_ring=True, inner=0, outer=args.r_outer, custom_mask=ps_mask)

    f_ary, df_rho_div_f_ary = load_psf()
    
    if args.diffuse == "ModelO":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_mO_pibrem, temp_mO_ics]
        # priors_poiss = [[-3., 0.001, 0.001, 0.001, 6., 1.], 
        #                 [1., 1.5, 1.5, 1.5, 12., 6.]]
        # params_log_poiss = [1, 0, 0, 0, 0, 0]
        priors_poiss = [[0., 0.001, 0.001, 0.001, 6., 1.], 
                        [2., 1.5, 1.5, 1.5, 12., 6.]]
        params_log_poiss = [0, 0, 0, 0, 0, 0]
    elif args.diffuse == "p6":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_dif]
        # priors_poiss = [[-3., 0.001, 0.001, 0.001, 10.], 
        #                 [1., 1.5, 1.5, 1.5, 20.]]
        # params_log_poiss = [1, 0, 0, 0, 0]
        priors_poiss = [[0., 0.001, 0.001, 0.001, 10.], 
                        [2., 1.5, 1.5, 1.5, 20.]]
        params_log_poiss = [0, 0, 0, 0, 0]
    else:
        raise NotImplementedError
    
    rescale = (fermi_exp / np.mean(fermi_exp))
    temps_ps = [temp_gce / rescale, temp_dsk / rescale]
    # priors_ps = [[-6., 10.0, 1.1, -10.0, 5.0, 0.1, -6., 10.0, 1.1, -10.0, 5.0, 0.1], 
                # [1., 20.0, 1.99, 1.99, 50.0, 4.99, 1., 20.0, 1.99, 1.99, 50.0, 4.99]]
    priors_ps = [[0., 10.0, 1.1, -10.0, 5.0, 0.1, 0., 10.0, 1.1, -10.0, 5.0, 0.1], 
                [2., 20.0, 1.99, 1.99, 50.0, 4.99, 2., 20.0, 1.99, 1.99, 50.0, 4.99]]

    # param_log = np.array(params_log_poiss + [1, 0, 0, 0, 0, 0, 1 ,0 ,0 ,0 ,0, 0])
    param_log = np.array(params_log_poiss + [0, 0, 0, 0, 0, 0, 1 ,0 ,0 ,0 ,0, 0])

    param_log = param_log.astype(np.bool)

    npr = NPRegression(temps_poiss, temps_ps, priors_poiss, fermi_data, priors_ps, f_ary, df_rho_div_f_ary, roi_mask=roi_mask, param_log=param_log, transform_prior_on_s=args.transform_prior_on_s)

    # Sample using chosen sampler
    if args.sampler == "dynesty":
        samples, samples_thinned_flattened = npr.run_dynesty(nlive=args.n_live, n_cpus=args.n_cpus)
    else:
        raise NotImplementedError
    
    # Save samples
    np.savez(args.save_dir + args.sample_name + "_samples.npz", samples=samples, samples_thinned_flattened=samples_thinned_flattened)
