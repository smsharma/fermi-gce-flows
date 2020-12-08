import numpy as np


def dnds(s, theta):

    a, n1, n2, n3, sb1, sb2 = theta
    dnds = a * np.where(s < sb2, (s / sb2) ** (-n3), np.where((s >= sb2) * (s < sb1), (s / sb2) ** (-n2), (sb1 / sb2) ** (-n2) * (s / sb1) ** (-n1)))
    return dnds
