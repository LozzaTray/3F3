"""file defining various useful probability density functions"""
import numpy as np


def n_pdf(x, mu=0., sigma=1.):
    """normal (gaussian) pdf"""
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / (2 * (sigma ** 2)))
    return y

def uniform_pdf(x, a=0.0, b=1.0):
    """Uniform distribution for range [a,b]"""
    if(a > b):
        raise ValueError("Upper limit of range must be greater than lower lim")

    # check array
    if(isinstance(x, np.ndarray)):
        return [uniform_pdf(x_elem) for x_elem in x]
    
    if (x >= a and x <= b):
        return 1 / (b-a)
    return 0