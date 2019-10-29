import numpy as np
import matplotlib.pyplot as plt


def n_pdf(x, mu=0., sigma=1.):
    """normal (gaussian) pdf"""
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / (2 * (sigma ** 2)))
    return y


def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""

    def ksd(x_axis):

        prob = [n_pdf(x_i, data, width) for x_i in x_axis]
        pdf = [np.average(pr) for pr in prob]  # each row is one x value
        return np.array(pdf)
    return ksd


# Plot normal distribution
fig, ax = plt.subplots(2)
x = np.random.randn(1000)  # randn is standard normal distribution
ax[0].hist(x, bins=30)  # number of bins
ks_density = ksdensity(x, width=0.4)
x_values = np.linspace(-5., 5., 100)
ax[1].plot(x_values, ks_density(x_values))

# Plot uniform distribution
fig2, ax2 = plt.subplots(2)
x = np.random.rand(1000)  # uniform distribution [0-1]
ax2[0].hist(x, bins=20)
ks_density = ksdensity(x, width=0.2)
x_values = np.linspace(-1., 2., 100)
ax2[1].plot(x_values, ks_density(x_values))

# Show plots
plt.show()
