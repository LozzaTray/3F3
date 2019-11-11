import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import n_pdf

N = 1000
bins = 30

# Plot normal distribution
fig = plt.figure()
x = np.random.randn(1000)  # randn is standard normal distribution
x_values = np.linspace(-5., 5., 100)

ks_density = ksdensity(x, width=0.4)
scale = lambda el: el * N / bins


plt.title('Histogram of normal distribution (N={})'.format(N))
plt.hist(x, bins=bins, density=True, label="normalised frequency")
plt.plot(x_values, n_pdf(x_values), linestyle='dashed', label="gaussian pdf")
plt.xlabel('x')
plt.ylabel('Normalised frequency')
plt.legend()

plt.show()