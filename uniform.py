import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import uniform_pdf

N = 1000
bins = 30

# Plot normal distribution
fig = plt.figure()
x = np.random.rand(1000)  # uniform dist [0,1]
x_values = np.linspace(-0.2, 1.2, 1000)

ks_density = ksdensity(x, width=0.4)
scale = lambda el: el * N / bins


plt.title('Histogram of uniform distribution (N={})'.format(N))
plt.hist(x, bins=bins, density=True, label="normalised frequency")
plt.plot(x_values, uniform_pdf(x_values), linestyle='dashed', label="gaussian pdf")
plt.xlabel('x')
plt.ylabel('Normalised frequency')
plt.legend()

plt.show()