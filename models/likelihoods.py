import sys

sys.path.append("./")

import torch
from models.scd import dnds_torch as dnds


def log_like_np(pt_sum_compressed, theta, npt_compressed, data, f_ary, df_rho_div_f_ary):
    """ Organize and combine non-Poissonian likelihoods across multiple templates
    """

    k_max = torch.max(data) + 1
    npixROI = int(pt_sum_compressed.size(0))

    x_m_ary = torch.zeros((npixROI, int(k_max) + 1), dtype=torch.float64)
    x_m_sum = torch.zeros(npixROI, dtype=torch.float64)

    s_ary = torch.logspace(-2, 2, 1000, dtype=torch.float64)

    for i in torch.arange(len(theta)):
        dnds_ary = dnds(s_ary, theta[i])

        x_m_ary_out, x_m_sum_out = return_x_m(f_ary, df_rho_div_f_ary, npt_compressed[i], data, s_ary, dnds_ary)
        x_m_ary += x_m_ary_out
        x_m_sum += x_m_sum_out

    return log_like_internal(pt_sum_compressed, data, x_m_ary, x_m_sum)


def log_like_internal(pt_sum_compressed, data, x_m_ary, x_m_sum):
    """ Non-Poissonian likelihood for single template, given x_m and x_m_sum
    """

    k_max = torch.max(data) + 1
    npixROI = int(pt_sum_compressed.size(0))

    f0_ary = -(pt_sum_compressed + x_m_sum)
    f1_ary = pt_sum_compressed + x_m_ary[:, 1]

    pk_ary = torch.zeros((npixROI, int(k_max) + 1), dtype=torch.float64)

    pk_ary[:, 0] = torch.exp(f0_ary)
    pk_ary[:, 1] = pk_ary[:, 0].clone() * f1_ary

    for k in torch.arange(2, k_max + 1):
        k = k.long()
        n = torch.arange(k - 1)
        pk_ary[:, k] = torch.sum((k - n) / k.float() * x_m_ary[:, k - n] * pk_ary[:, n].clone(), axis=1) + f1_ary * pk_ary[:, k - 1].clone() / k.float()

    pk_dat_ary = (pk_ary[torch.arange(npixROI), data.long()]).double()

    return torch.log(pk_dat_ary)


def return_x_m(f_ary, df_rho_div_f_ary, npt_compressed, data, s_ary, dnds_ary):
    """ Dedicated calculation of x_m and x_m_sum
    """

    k_max = torch.max(data) + 1
    m_ary = torch.arange(k_max + 1, dtype=torch.float64)
    gamma_ary = torch.exp(torch.lgamma(m_ary + 1))

    x_m_ary = df_rho_div_f_ary[:, None] * f_ary[:, None] * torch.trapz(((dnds_ary * (-torch.ger(f_ary, s_ary)).exp())[:, :, None] * (torch.ger(f_ary, s_ary)[:, :, None]).pow(m_ary) / gamma_ary), s_ary, axis=1)
    x_m_ary = torch.sum(x_m_ary, axis=0)

    x_m_ary = torch.ger(npt_compressed, x_m_ary)

    # Get x_m_sum_ary array

    x_m_sum_ary = torch.sum((df_rho_div_f_ary * f_ary)[:, None] * torch.trapz(dnds_ary, s_ary), axis=0)
    x_m_sum_ary = torch.sum(x_m_sum_ary, axis=0)

    x_m_sum_ary = npt_compressed * x_m_sum_ary - x_m_ary[:, 0]

    return x_m_ary, x_m_sum_ary
