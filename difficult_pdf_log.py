import numpy as np
import matplotlib.pyplot as plt
from pdfs import n_pdf
from ksdensity import ksdensity
from functools import partial


def F_inverse(x, a):
    "inverse cdf for exponential function"
    #((a**2)/2) * np.exp(-(2*u)/(a**2)) # pdf
    return - ((a**2)/(2)) * np.log(1 - x)


def inverse_cdf_sample(N, f):
    """
    Inverse cdf method
        N - number of samples
        f - inverse cdf function
    """
    uniform = np.random.rand(N)
    sampled = f(uniform)
    return sampled


def sample_gaussian(N, mean=0, variance=1):
    """Generate N samples from a generic gaussian"""
    std_dev = np.sqrt(variance)
    originals = np.random.randn(N)
    transformed = (originals * std_dev) + mean
    return transformed


N = 1000
a_array = np.linspace(0.1, 10, 10)
g = []
sigma = 0.4
x_values = np.linspace(-1., 1., 1000)

fig = plt.figure()
plt.title('Log of kernel smoothed X distribution (N={})'.format(N))

for a in a_array:

    f = partial(F_inverse, a=a)
    u_array = inverse_cdf_sample(N, f)
    x_array = []

    for u in u_array:
        x_val = sample_gaussian(1, variance=u)
        x_array.extend(x_val)

    ks_density = ksdensity(x_array, width=sigma)
    plt.plot(x_values, np.log(ks_density(x_values)), label="alpha={}".format(a))
    g.extend(np.log(ks_density(np.zeros(1))))

plt.xlabel('x')
plt.ylabel('ln(probability density)')
plt.legend()
plt.grid()

img_dir = "/mnt/c/Users/ltray/Documents/Cambridge/3F3/img/"
plt.savefig(img_dir + "difficult_log.png")

fig = plt.figure()
sigma = np.exp(-2*np.array(g))/np.pi
plt.plot(sigma, np.log(a_array))
plt.ylabel("ln(alpha)")
plt.xlabel("sigma")
plt.grid()

plt.show()