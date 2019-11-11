import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import n_pdf

N = 1000
bins = 30

# Plot normal distribution
fig, ax = plt.subplots(2, sharex='col')
x = np.random.randn(1000)  # randn is standard normal distribution
x_values = np.linspace(-5., 5., 100)

ks_density = ksdensity(x, width=0.4)
scale = lambda el: el * N / bins

ax[0].set_title('Histogram of normal distribution (N={})'.format(N))
ax[0].hist(x, bins=bins, density=True)  # number of bins
ax[0].plot(x_values, n_pdf(x_values), linestyle='dashed')

ax[1].set_title('Smoothed curve')
ax[1].plot(x_values, ks_density(x_values))
ax[1].plot(x_values, n_pdf(x_values), linestyle='dashed')

# Plot uniform distribution
fig2, ax2 = plt.subplots(2)
x = np.random.rand(1000)  # uniform distribution [0-1]
ax2[0].hist(x, bins=20)
ks_density = ksdensity(x, width=0.2)
x_values = np.linspace(-1., 2., 100)
ax2[1].plot(x_values, ks_density(x_values))

# Show plots
plt.show()
