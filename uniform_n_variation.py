import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import uniform_pdf

N = 1000
J = 30

img_dir = "/mnt/c/Users/ltray/Documents/Cambridge/3F3/img/"

x_values = np.linspace(-0.2, 1.2, 1000)

fig, ax = plt.subplots(3, sharex='col')
plt.title('Uniform distribution histograms')

for i in range(3):
    axis = ax[i]
    N = 10 ** (2+i)
    x = np.random.rand(N)  # uniform dist [0,1]
    mu = N/J
    sigma = np.sqrt(N*(J-1)) / J

    axis.set_title('N={}'.format(N))
    axis.hist(x, bins=J)
    
    axis.axhline(mu, label="mean", linestyle='dashed', color='#ffa500')
    axis.axhline(mu - 3*sigma, label="-3 sigma", linestyle='dashed', color='0.6')
    axis.axhline(mu + 3*sigma, label="+3 sigma", linestyle='dashed', color='0.6')
    axis.legend()

plt.savefig(img_dir + "uniform_n_hist.png")
plt.show()