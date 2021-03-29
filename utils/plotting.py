from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import warnings
import matplotlib.cbook
import healpy as hp
from getdist import plots, MCSamples

from utils.plot_params import params
from models.scd import dnds

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

pylab.rcParams.update(params)
cols_default = plt.rcParams['axes.prop_cycle'].by_key()['color']

def dnds_conv(s_ary, theta, ps_temp, roi_counts_normalize, roi_normalize):
    dnds_ary = dnds(s_ary, [1] + list(theta[1:]))
    A = theta[0] / np.trapz(s_ary * dnds_ary, s_ary) / np.mean(ps_temp[~roi_counts_normalize]) ** 2 * np.mean(ps_temp[~roi_normalize])
    dnds_ary = dnds(s_ary, [A] + list(theta[1:]))
    return dnds_ary

def make_plot(posterior, x_test, x_data_test=None, theta_test=None, roi_normalize=None, roi_sim=None, roi_counts_normalize=None, is_data=False, signal_injection=False, figsize=(25, 18), save_filename=None, nptf=False, n_samples=10000, nside=128, coeff_ary=None, temps_dict=None):

    # Extract templates and labels
    n = SimpleNamespace(**temps_dict)
    fermi_exp, temps_ps, temps_ps_sim, ps_labels, temps_poiss, temps_poiss_sim, poiss_labels = n.fermi_exp, n.temps_ps, n.temps_ps_sim, n.ps_labels, n.temps_poiss, n.temps_poiss_sim, n.poiss_labels
    
    pixarea = hp.nside2pixarea(nside, degrees=False)
    pixarea_deg = hp.nside2pixarea(nside, degrees=True)

    # Set up plot

    n_datasets = x_test.shape[0]
    nrows = n_datasets

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=4, width_ratios=[1,2,1,1])

    ax = [[None] * 4] * nrows

    # Go row by row and plot

    for i_r in range(nrows):

        x_o = x_test[i_r]
        
        if x_data_test is not None:
            x_d = x_data_test
        else:
            x_d = x_o[:,:-2]
                
        if not is_data:
            theta_truth = theta_test[i_r]
        
        if nptf:
            posterior_samples = posterior[i_r]
        else:
            posterior_samples = posterior.sample((n_samples,), x=x_o)
            posterior_samples = posterior_samples.detach().numpy()
        
        # Counts and flux arrays
        s_f_conv = np.mean(fermi_exp[~roi_counts_normalize])
        s_ary = np.logspace(-1, 2, 100)
        f_ary = np.logspace(-1, 2, 100) / s_f_conv
        
        ## 1. Source count distributions plot

        ax[i_r][0] = fig.add_subplot(gs[i_r,0])
        
        for idx_ps, i_param_ps in enumerate([6, 12]):

            dnds_ary = np.array([dnds_conv(s_ary, theta, temps_ps[idx_ps], roi_counts_normalize,roi_normalize) for theta in posterior_samples[:,i_param_ps:i_param_ps+6]])
            dnds_ary *= s_f_conv / pixarea_deg
            ax[i_r][0].plot(f_ary, np.median(f_ary ** 2 * dnds_ary, axis=0), color=cols_default[idx_ps], lw=0.8)
            ax[i_r][0].fill_between(f_ary, np.percentile(f_ary ** 2 * dnds_ary, [16], axis=0)[0], np.percentile(f_ary ** 2 * dnds_ary, [84], axis=0)[0], alpha=0.2, color=cols_default[idx_ps], label=ps_labels[idx_ps])
            ax[i_r][0].fill_between(f_ary, np.percentile(f_ary ** 2 * dnds_ary, [2.5], axis=0)[0], np.percentile(f_ary ** 2 * dnds_ary, [97.5], axis=0)[0], alpha=0.1, color=cols_default[idx_ps])

            if not is_data:
                ax[i_r][0].plot(f_ary, f_ary ** 2 * dnds_conv(s_ary, theta_truth[i_param_ps:i_param_ps+6], temps_ps[idx_ps], roi_counts_normalize, roi_normalize) * (s_f_conv / pixarea_deg), color=cols_default[idx_ps], ls='dotted', label=ps_labels[idx_ps] + " truth")

        ax[i_r][0].set_xscale("log")
        ax[i_r][0].set_yscale("log")

        ax[i_r][0].set_ylim(1e-13, 2e-10)
        ax[i_r][0].set_xlim(3e-12, 1e-9)
                    
        ax[i_r][0].set_ylabel(r"$F^2\,\mathrm{d}N/\mathrm{d}F$\,[ph\,cm$^{-2}$\,s$^{-1}$\,sr$^{-1}$]")

        if i_r == nrows - 1:
            ax[i_r][0].set_xlabel(r"$F$\,[ph\,cm$^{-2}$\,s$^{-1}$]")

        if i_r == 0:
            ax[i_r][0].set_title(r"\bf{Source-count distribution}")

        ax[i_r][0].legend(fontsize=14)

        ## Fluxes plot, all templates except diffuse

        ax[i_r][2] = fig.add_subplot(gs[i_r,1])
        ax2_max = 4.5

        bins = np.linspace(0., 5, 60)
        hist_kwargs = {'bins':bins, 'alpha':0.9, 'histtype':'step', 'lw':1.5, 'density':True}
        divide_by = 1e-7

        mean_counts_roi_post = np.zeros(len(posterior_samples))

        for i_temp_ps, idx_ps in enumerate([6,12]):
            
            mean_counts_roi = posterior_samples[:, idx_ps] * np.mean(temps_ps[i_temp_ps][~roi_normalize]) / np.mean(temps_ps[i_temp_ps][~roi_counts_normalize])

            mean_counts_roi_post += mean_counts_roi

            ax[i_r][2].hist(mean_counts_roi / np.mean(fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[i_temp_ps], label=ps_labels[i_temp_ps],  **hist_kwargs)
            if not is_data:
                ax[i_r][2].axvline(theta_truth[idx_ps] * np.mean(temps_ps_sim[i_temp_ps][~roi_normalize]) / np.mean(temps_ps_sim[i_temp_ps][~roi_counts_normalize]) / np.mean(fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[i_temp_ps], ls='dotted')

        for i_temp_poiss in range(len(temps_poiss)):

            mean_counts_roi = posterior_samples[:, 1 + i_temp_poiss] * np.mean(temps_poiss[i_temp_poiss][~roi_normalize])
            mean_counts_roi_post += mean_counts_roi

            if i_temp_poiss in [2,3,4]:
                continue

            ax[i_r][2].hist(mean_counts_roi / np.mean(fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[len(temps_ps) + i_temp_poiss], label=poiss_labels[i_temp_poiss],  **hist_kwargs)
            if not is_data:
                ax[i_r][2].axvline(theta_truth[1 + i_temp_poiss] * np.mean(temps_poiss_sim[1 + i_temp_poiss][~roi_normalize] / fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[len(temps_ps) + i_temp_poiss], ls='dotted')

        i_temp_ps = 0

        mean_counts_roi = posterior_samples[:, 0] * np.mean(temps_ps[i_temp_ps][~roi_normalize]) / np.mean(temps_ps[i_temp_ps][~roi_counts_normalize])
        mean_counts_roi_post += mean_counts_roi

        ax[i_r][2].hist(mean_counts_roi / np.mean(fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[8], label='gce',  **hist_kwargs)
        if not is_data:
            ax[i_r][2].axvline(theta_truth[0] * np.mean(temps_ps_sim[i_temp_ps][~roi_normalize]) / np.mean(temps_ps_sim[i_temp_ps][~roi_counts_normalize]) / np.mean(fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[8], ls='dotted')
        
        if i_r == nrows - 1:
            ax[i_r][2].set_xlabel(r"Flux\,[$10^{-7}$\,ph\,cm$^{-2}$\,s$^{-1}$\,sr$^{-1}$]")
        if i_r == 0:
            ax[i_r][2].set_title(r"\bf{Fluxes}")

        ax[i_r][2].set_xlim(0, ax2_max - 0.05)
        ax[i_r][2].set_ylim(0, 3.5)

        ## Fluxes plot, diffuse templates

        ax[i_r][3] = fig.add_subplot(gs[i_r,2])
        ax3_min = 2.
        ax3_max = 25.

        bins = np.linspace(ax3_min, ax3_max, np.int(60))
        hist_kwargs.update(bins=bins)
        divide_by = 1e-7

        for i_temp_poiss in [3,4]:

            ax[i_r][3].hist(posterior_samples[:, 1 + i_temp_poiss] * np.mean(temps_poiss[i_temp_poiss][~roi_normalize] / fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[len(temps_ps) + i_temp_poiss], label=poiss_labels[i_temp_poiss], **hist_kwargs)
            if not is_data:
                ax[i_r][3].axvline(theta_truth[1 + i_temp_poiss] * np.mean(temps_poiss_sim[1 + i_temp_poiss][~roi_normalize] / fermi_exp[~roi_normalize]) / pixarea / divide_by, color=cols_default[len(temps_ps) + i_temp_poiss], ls='dotted')

        # hide the spines between ax and ax2
        ax[i_r][2].spines['right'].set_visible(False)
        ax[i_r][3].spines['left'].set_visible(False)
        ax[i_r][2].tick_params(labelright=False, labelleft=False, right=False, which='both')
        ax[i_r][3].tick_params(labelleft=False, left=False, which='both')  

        ax[i_r][3].set_xlim(ax3_min + .05, ax3_max)
        ax[i_r][3].set_ylim(0, 2)

        # Create dividers between the two flux plots

        handles_2, labels_2 = ax[i_r][2].get_legend_handles_labels()
        handles_3, labels_3 = ax[i_r][3].get_legend_handles_labels()

        ax[i_r][2].legend(handles_2 + handles_3, labels_2 + labels_3, fontsize=14, ncol=2)

        d = .02 
        kwargs = dict(transform=ax[i_r][2].transAxes, color='k', lw=1.8, clip_on=False)
        ax[i_r][2].plot((1, 1), (1 - d, 1 + d), **kwargs)
        ax[i_r][2].plot((1, 1), (-d, +d), **kwargs)

        kwargs.update(transform=ax[i_r][3].transAxes)
        ax[i_r][3].plot((0, 0), (1 - d, 1 + d), **kwargs)
        ax[i_r][3].plot((0, 0), (-d, +d), **kwargs) 
    
        ## Flux fractions plot

        ax[i_r][1] = fig.add_subplot(gs[i_r,-1])

        if nptf:
            x_embedded = np.zeros(hp.nside2npix(128))
            x_embedded[np.where(~roi_sim)] = x_d[0]
            mean_roi_counts = np.mean(x_embedded[~roi_counts_normalize])
        else:
            x_embedded = np.zeros(hp.nside2npix(128))
            x_embedded[np.where(~roi_sim)] = x_d[0]
            mean_roi_counts = np.mean(x_embedded[~roi_counts_normalize])
            
        ax[i_r][0].axvline(np.sqrt(mean_roi_counts) / np.mean(fermi_exp[~roi_normalize]), lw=1, ls='dotted', color='grey')

        fraction_multiplier = 100 * np.mean(temps_ps[0][~roi_normalize]) / np.mean(temps_ps[0][~roi_counts_normalize]) / mean_counts_roi_post

        g = plots.get_single_plotter()
        samples = MCSamples(samples=np.transpose(np.array([posterior_samples[:, 0] * fraction_multiplier, posterior_samples[:, 6] * fraction_multiplier])),names = ['DM','PS'], labels = ['DM','PS'])
        g.plot_2d(samples, 'DM', 'PS', filled=True, alphas=[0.5], ax=ax[i_r][1], colors=[cols_default[0]])
        g.plot_2d(samples, 'DM', 'PS', filled=False, ax=ax[i_r][1], colors=['k'], lws=[1.2])

        # TODO: Take span depending on mean roi counts rather than posterior
        
        if signal_injection:
            if i_r == 0:
                theta_dm_baseline = np.median(posterior_samples[:, 0])
                theta_ps_baseline = np.median(posterior_samples[:, 6])
            else:
                ax[i_r][1].axvline((theta_dm_baseline + coeff_ary[i_r]) * np.median(fraction_multiplier), color='k', ls='dotted')
                ax[i_r][1].axhline((theta_ps_baseline) * np.median(fraction_multiplier), color='k', ls='dotted')

        if not is_data:
            ax[i_r][1].axvline(theta_truth[0] * np.median(fraction_multiplier), color='k', ls='dotted')
            ax[i_r][1].axhline(theta_truth[6] * np.median(fraction_multiplier), color='k', ls='dotted')

        ax[i_r][1].set_xlim(0., 15.)
        ax[i_r][1].set_ylim(0., 15.)

        ax[i_r][1].set_ylabel(r"PS\,[\%]", fontsize=17.5)
        if i_r == nrows - 1:
            ax[i_r][1].set_xlabel(r"DM\,[\%]", fontsize=17.5)
        else:
            ax[i_r][1].set_xlabel(None, fontsize=17.5)
        if i_r == 0:
            ax[i_r][1].set_title(r"\bf{Flux fractions}")

        ax[i_r][1].tick_params(axis='x', labelsize=17.5)
        ax[i_r][1].tick_params(axis='y', labelsize=17.5)

    # Optionally save plot

    if save_filename is not None:
        plt.tight_layout()
        fig.savefig(save_filename,bbox_inches='tight',pad_inches=0.1)

def make_signal_injection_plot(posterior, x_test, x_data_test=None, theta_test=None, roi_normalize=None, roi_sim=None, roi_counts_normalize=None, is_data=False, signal_injection=False, figsize=(25, 18), save_filename=None, nptf=False, n_samples=10000, nside=128, coeff_ary=None, temps_dict=None):

    # Extract templates and labels
    n = SimpleNamespace(**temps_dict)
    fermi_exp, temps_ps, temps_ps_sim, ps_labels, temps_poiss, temps_poiss_sim, poiss_labels = n.fermi_exp, n.temps_ps, n.temps_ps_sim, n.ps_labels, n.temps_poiss, n.temps_poiss_sim, n.poiss_labels
    
    pixarea = hp.nside2pixarea(nside, degrees=False)
    pixarea_deg = hp.nside2pixarea(nside, degrees=True)

    # Set up plot

    n_datasets = x_test.shape[0]
    nrows = n_datasets

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=n_datasets, width_ratios=n_datasets * [1])

    ax = [None] * n_datasets

    # Go row by row and plot

    for i_r in range(nrows):

        x_o = x_test[i_r]
        
        if x_data_test is not None:
            x_d = x_data_test
        else:
            x_d = x_o[:,:-2]
                
        if not is_data:
            theta_truth = theta_test[i_r]
        
        if nptf:
            posterior_samples = posterior[i_r]
        else:
            posterior_samples = posterior.sample((n_samples,), x=x_o)
            posterior_samples = posterior_samples.detach().numpy()
        
        # Counts and flux arrays
        s_f_conv = np.mean(fermi_exp[~roi_counts_normalize])
        s_ary = np.logspace(-1, 2, 100)
        f_ary = np.logspace(-1, 2, 100) / s_f_conv
        
        mean_counts_roi_post = np.zeros(len(posterior_samples))

        for i_temp_ps, idx_ps in enumerate([6,12]):
            
            mean_counts_roi = posterior_samples[:, idx_ps] * np.mean(temps_ps[i_temp_ps][~roi_normalize]) / np.mean(temps_ps[i_temp_ps][~roi_counts_normalize])

            mean_counts_roi_post += mean_counts_roi

        for i_temp_poiss in range(len(temps_poiss)):

            mean_counts_roi = posterior_samples[:, 1 + i_temp_poiss] * np.mean(temps_poiss[i_temp_poiss][~roi_normalize])
            mean_counts_roi_post += mean_counts_roi

        i_temp_ps = 0

        mean_counts_roi = posterior_samples[:, 0] * np.mean(temps_ps[i_temp_ps][~roi_normalize]) / np.mean(temps_ps[i_temp_ps][~roi_counts_normalize])
        mean_counts_roi_post += mean_counts_roi

        ## Flux fractions plot

        ax[i_r] = fig.add_subplot(gs[i_r])

        fraction_multiplier = 100 * np.mean(temps_ps[0][~roi_normalize]) / np.mean(temps_ps[0][~roi_counts_normalize]) / mean_counts_roi_post

        g = plots.get_single_plotter()
        samples = MCSamples(samples=np.transpose(np.array([posterior_samples[:, 0] * fraction_multiplier, posterior_samples[:, 6] * fraction_multiplier])),names = ['DM','PS'], labels = ['DM','PS'])
        g.plot_2d(samples, 'DM', 'PS', filled=True, alphas=[0.5], ax=ax[i_r], colors=[cols_default[0]])
        g.plot_2d(samples, 'DM', 'PS', filled=False, ax=ax[i_r], colors=['k'], lws=[1.2])

        # TODO: Take span depending on mean roi counts rather than posterior
        
        if signal_injection:
            if i_r == 0:
                theta_dm_baseline = np.median(posterior_samples[:, 0])
                theta_ps_baseline = np.median(posterior_samples[:, 6])
            else:
                ax[i_r].axvline((theta_dm_baseline + coeff_ary[i_r]) * np.median(fraction_multiplier), color='k', ls='dotted')
                ax[i_r].axhline((theta_ps_baseline) * np.median(fraction_multiplier), color='k', ls='dotted')

        if not is_data:
            ax[i_r].axvline(theta_truth[0] * np.median(fraction_multiplier), color='k', ls='dotted')
            ax[i_r].axhline(theta_truth[6] * np.median(fraction_multiplier), color='k', ls='dotted')

        ax[i_r].set_xlim(0., 15.)
        ax[i_r].set_ylim(0., 15.)

        ax[i_r].set_ylabel(r"PS\,[\%]", fontsize=17.5)
        if i_r == nrows - 1:
            ax[i_r].set_xlabel(r"DM\,[\%]", fontsize=17.5)
        else:
            ax[i_r].set_xlabel(None, fontsize=17.5)
        if i_r == 0:
            ax[i_r].set_title(r"\bf{Flux fractions}")

        ax[i_r].tick_params(axis='x', labelsize=17.5)
        ax[i_r].tick_params(axis='y', labelsize=17.5)

    # Optionally save plot

    if save_filename is not None:
        plt.tight_layout()
        fig.savefig(save_filename,bbox_inches='tight',pad_inches=0.1)