import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import n_pdf

N = 1000
sigma = 0.4
bins = 30

img_dir = "/mnt/c/Users/ltray/Documents/Cambridge/3F3/img/"

# Plot normal distribution
x = np.random.randn(1000)  # randn is standard normal distribution
x_values = np.linspace(-5., 5., 100)

ks_density = ksdensity(x, width=0.4)
scale = lambda el: el * N / bins

fig = plt.figure()

plt.title('Histogram of gaussian distribution (N={})'.format(N))
plt.hist(x, bins=bins, density=True, label="normalised frequency")
plt.plot(x_values, n_pdf(x_values), linestyle='dashed', label="gaussian pdf")
plt.xlabel('x')
plt.ylabel('Normalised frequency')
plt.legend()

#plt.savefig(img_dir + "gaussian_hist.png")

fig = plt.figure()
plt.title('Kernel smoothed gaussian distribution (N={}, sigma={})'.format(N, sigma))
ks_density = ksdensity(x, width=sigma)
plt.plot(x_values, ks_density(x_values) / sigma, label="kernel smoothed")
plt.plot(x_values, n_pdf(x_values), linestyle='dashed', label="gaussian pdf")
plt.xlabel('x')
plt.ylabel('probability density')
plt.legend()

#plt.savefig(img_dir + "gaussian_ksd.png")
plt.show()