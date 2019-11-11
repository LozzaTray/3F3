import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import uniform_pdf

N = 1000
sigma = 0.2
bins = 30

img_dir = "/mnt/c/Users/ltray/Documents/Cambridge/3F3/img/"

# Plot normal distribution
x = np.random.rand(1000)  # uniform dist [0,1]
x_values = np.linspace(-0.2, 1.2, 1000)

fig = plt.figure()
plt.title('Histogram of uniform distribution (N={})'.format(N))
plt.hist(x, bins=bins, density=True, label="normalised frequency")
plt.plot(x_values, uniform_pdf(x_values), linestyle='dashed', label="uniform pdf")
plt.xlabel('x')
plt.ylabel('Normalised frequency')
plt.legend()

plt.savefig(img_dir + "uniform_hist.png")

fig = plt.figure()
plt.title('Kernel smoothed uniform distribution (N={}, sigma={})'.format(N, sigma))
ks_density = ksdensity(x, width=sigma)
x_values = np.linspace(-1., 2., 100)
plt.plot(x_values, ks_density(x_values) / sigma, label="kernel smoothed")
plt.plot(x_values, uniform_pdf(x_values), linestyle='dashed', label="uniform pdf")
plt.xlabel('x')
plt.ylabel('probability density')
plt.legend()

plt.savefig(img_dir + "uniform_ksd.png")
plt.show()