import numpy as np
import torch


def dnds(s, theta):

    a, n1, n2, n3, sb1, sb2 = theta
    dnds = a * (sb2 / sb1) ** -n2 * np.where(s < sb2, (s / sb2) ** (-n3), np.where((s >= sb2) * (s < sb1), (s / sb2) ** (-n2), (sb1 / sb2) ** (-n2) * (s / sb1) ** (-n1)))
    return dnds

def dnds_torch(s, theta):

    a, n1, n2, n3, sb1, sb2 = theta
    dnds = a * (sb2 / sb1).pow(-n2) * torch.where(s < sb2, (s / sb2).pow(-n3), torch.where((s >= sb2) * (s < sb1), (s / sb2).pow(-n2), (sb1 / sb2).pow(-n2) * (s / sb1).pow(-n1)))
    return dnds
