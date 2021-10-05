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
    def __init__(self, temps_poiss, temps_ps, priors_poiss, data, priors_ps=None, f_ary=[1.], df_rho_div_f_ary=[1.], roi_mask=None, param_log=None, transform_prior_on_s=True, roi_counts_normalize=None):
        
        self.transform_prior_on_s = transform_prior_on_s
        
        if roi_counts_normalize is None:
            self.roi_counts_normalize = roi_mask
        else:
            self.roi_counts_normalize = roi_counts_normalize

        self.temps_poiss= temps_poiss
        self.temps_ps = temps_ps
        self.temps_ps_og = temps_ps
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

                s_ary = torch.logspace(-2, 2, 500)
                s_exp_temp = theta_ps[i_ps][0] * np.mean(self.temps_ps[i_ps]) / np.mean(self.temps_ps_og[i_ps][~self.roi_counts_normalize])
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
            sampler.run_nested(dlogz=0.1)
            
        # Draw posterior samples
        weights = np.exp(sampler.results['logwt'] - sampler.results['logz'][-1])
        samples_weighted = resample_equal(sampler.results.samples, weights)

        return samples_weighted

def load_psf(psf_type="king"):

    if psf_type == "king":

        # Define parameters that specify the Fermi-LAT PSF at 2 GeV
        fcore = 0.748988248179
        score = 0.428653790656
        gcore = 7.82363229341
        stail = 0.715962650769
        gtail = 3.61883748683
        spe = 0.00456544262478

        # Define the full PSF in terms of two King functions
        def king_fn(x, sigma, gamma):
            return 1./(2.*np.pi*sigma**2.)*(1.-1./gamma)*(1.+(x**2./(2.*gamma*sigma**2.)))**(-gamma)

        def Fermi_PSF(r):
            return fcore*king_fn(r/spe,score,gcore) + (1-fcore)*king_fn(r/spe,stail,gtail)

        # Modify the relevant parameters in pc_inst and then make or load the PSF
        pc_inst = PSFCorrection(delay_compute=True)
        pc_inst.psf_r_func = lambda r: Fermi_PSF(r)
        pc_inst.sample_psf_max = 10.*spe*(score+stail)/2.
        pc_inst.psf_samples = 10000
        pc_inst.psf_tag = 'Fermi_PSF_2GeV'
        pc_inst.make_or_load_psf_corr()

    elif psf_type == "gaussian":
        pc_inst = PSFCorrection(psf_sigma_deg=0.1812)
    else:
        raise NotImplementedError

    return pc_inst.f_ary, pc_inst.df_rho_div_f_ary

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_cpus", action="store", default=36, type=int)
    parser.add_argument("--n_live", action="store", default=500, type=int)
    parser.add_argument("--sample_name", action="store", default="ModelO_PS_only", type=str)
    parser.add_argument("--sampler", action="store", default="dynesty", type=str)
    parser.add_argument("--save_dir", action="store", default="data/nptfit_samples/", type=str)
    parser.add_argument("--r_outer", action="store", default=25., type=float)
    parser.add_argument("--r_outer_normalize", action="store", default=25., type=float)
    parser.add_argument("--ps_mask_type", action="store", default="0.8", type=str)
    parser.add_argument("--transform_prior_on_s", action="store", default=1, type=int)
    parser.add_argument("--diffuse", action="store", default="ModelO", type=str)
    parser.add_argument("--disk_type", action="store", default="thin", type=str)
    parser.add_argument("--i_mc", action="store", default=-1, type=int)
    parser.add_argument("--psf", action="store", default="king", type=str)
    parser.add_argument("--new_ps_priors", action="store", default=0, type=int)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Load templates
    # temp_gce = np.load("data/fermi_data/template_gce.npy") 
    temp_gce = get_NFW2_template(gamma=1.2)
    temp_dif = np.load("data/fermi_data/template_dif.npy")
    temp_psc = np.load("data/fermi_data/template_psc.npy")
    temp_iso = np.load("data/fermi_data/template_iso.npy")

    if args.disk_type == "thick":
        temp_dsk = np.load("data/fermi_data/template_dsk.npy")
    elif args.disk_type == "thin":
        temp_dsk = np.load("data/external/template_disk_r_s_5_z_s_0.3.npy")

    temp_bub = np.load("data/fermi_data/template_bub.npy")

    # Load Model O templates
    temp_mO_pibrem = np.load('data/fermi_data/ModelO_r25_q1_pibrem.npy')
    temp_mO_ics = np.load('data/fermi_data/ModelO_r25_q1_ics.npy')

    # Load Model A templates
    temp_mA_pibrem = hp.ud_grade(np.load('data/external/template_Api.npy'), nside_out=128, power=-2)
    temp_mA_ics = hp.ud_grade(np.load('data/external/template_Aic.npy'), nside_out=128, power=-2)
    
    # Load Model F templates
    temp_mF_pibrem = hp.ud_grade(np.load('data/external/template_Fpi.npy'), nside_out=128, power=-2)
    temp_mF_ics = hp.ud_grade(np.load('data/external/template_Fic.npy'), nside_out=128, power=-2)

    fermi_exp = np.load("data/fermi_data/fermidata_exposure.npy")
    fermi_data = np.load("data/fermi_data/fermidata_counts.npy")
    
    if args.i_mc != -1:

        # Create MC mask
        hp_mask_nside1 = cm.make_mask_total(nside=1, band_mask = True, band_mask_range = 0, mask_ring = True, inner = 0, outer = 25)
        mc_mask = hp.ud_grade(hp_mask_nside1, 128)

        x = np.load("data/samples/x_{}.npy".format(args.sample_name))[args.i_mc]
        fermi_data = np.zeros(hp.nside2npix(128))
        fermi_data[np.where(~mc_mask)] = x[0]

    if args.ps_mask_type == "0.8":
        ps_mask = np.load("data/mask_3fgl_0p8deg.npy")
    elif args.ps_mask_type == "default":
        ps_mask = np.load("data/fermi_data/fermidata_pscmask.npy") == 1
    else:
        raise NotImplementedError

    roi_mask = cm.make_mask_total(nside=128, band_mask = True, band_mask_range=2, mask_ring=True, inner=0, outer=args.r_outer, custom_mask=ps_mask)

    roi_counts_normalize = cm.make_mask_total(nside=128, band_mask = True, band_mask_range = 2, mask_ring = True, inner = 0, outer=args.r_outer_normalize)

    f_ary, df_rho_div_f_ary = load_psf(psf_type=args.psf)
    
    if args.diffuse == "ModelO":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_mO_pibrem, temp_mO_ics]
        priors_poiss = [[0., 0.001, 0.001, 0.001, 6., 1.], 
                        [2., 1.5, 1.5, 1.5, 12., 6.]]
        params_log_poiss = [0, 0, 0, 0, 0, 0]
    elif args.diffuse == "ModelA":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_mA_pibrem, temp_mA_ics]
        priors_poiss = [[0., 0.001, 0.001, 0.001, 6., 1.], 
                        [2., 1.5, 1.5, 1.5, 12., 6.]]
        params_log_poiss = [0, 0, 0, 0, 0, 0]
    elif args.diffuse == "ModelF":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_mF_pibrem, temp_mF_ics]
        priors_poiss = [[0., 0.001, 0.001, 0.001, 6., 1.], 
                        [2., 1.5, 1.5, 1.5, 12., 6.]]
        params_log_poiss = [0, 0, 0, 0, 0, 0]
    elif args.diffuse == "p6":
        temps_poiss = [temp_gce, temp_iso, temp_bub, temp_psc, temp_dif]
        priors_poiss = [[0., 0.001, 0.001, 0.001, 10.], 
                        [2., 1.5, 1.5, 1.5, 20.]]
        params_log_poiss = [0, 0, 0, 0, 0]
    else:
        raise NotImplementedError
    
    rescale = (fermi_exp / np.mean(fermi_exp))
    temps_ps = [temp_gce / rescale, temp_dsk / rescale]

    if args.new_ps_priors:
        priors_ps = [[0.001, 10.0, 1.1, -10.0, 1.0, 0.1, 0.001, 10.0, 1.1, -10.0, 1.0, 0.1], 
                    [2.5, 20.0, 1.99, 0.99, 30.0, 0.99, 2.5, 20.0, 1.99, 0.99, 30.0, 0.99]]
    else:
        priors_ps = [[0., 10.0, 1.1, -10.0, 5.0, 0.1, 0., 10.0, 1.1, -10.0, 5.0, 0.1], 
                    [2., 20.0, 1.99, 0.99, 50.0, 4.99, 2., 20.0, 1.99, 0.99, 50.0, 4.99]]

    param_log = np.array(params_log_poiss + [0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0, 0])

    param_log = param_log.astype(np.bool)

    npr = NPRegression(temps_poiss, temps_ps, priors_poiss, fermi_data, priors_ps, f_ary, df_rho_div_f_ary, roi_mask=roi_mask, param_log=param_log, transform_prior_on_s=args.transform_prior_on_s, roi_counts_normalize=roi_counts_normalize)

    # Sample using chosen sampler
    if args.sampler == "dynesty":
        samples_weighted = npr.run_dynesty(nlive=args.n_live, n_cpus=args.n_cpus)
    else:
        raise NotImplementedError
    
    # Save samples
    np.savez("{}/{}_{}_samples.npz".format(args.save_dir, args.sample_name, args.i_mc), samples_weighted=samples_weighted)
