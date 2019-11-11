"""file defining various useful probability density functions"""
import numpy as np


def n_pdf(x, mu=0., sigma=1.):
    """normal (gaussian) pdf"""
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / (2 * (sigma ** 2)))
    return y