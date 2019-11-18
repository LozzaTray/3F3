import numpy as np
import matplotlib.pyplot as plt
from ksdensity import ksdensity
from pdfs import n_pdf


def f_1(x, a, b):
    """f(x) = ax + b"""
    # check array
    if(isinstance(x, np.ndarray)):
        return [f_1(x_elem, a=a, b=b) for x_elem in x]

    return a*x + b


def f_2(x):
    """f(x) = x**2"""
    # check array
    if(isinstance(x, np.ndarray)):
        return [f_2(x_elem) for x_elem in x]

    return x**2


def pdf_2(x):
    """transformed pdf"""
    if(isinstance(x, np.ndarray)):
        return [pdf_2(x_elem) for x_elem in x]

    if (x < 0):
        return 0

    return (1 / (np.sqrt(2 * np.pi * x))) * np.exp(- x/2)


N=100000
bins=30
a=1.4
b=3

img_dir="/mnt/c/Users/ltray/Documents/Cambridge/3F3/img/"

# Plot normal distribution
x=np.random.randn(1000)  # randn is standard normal distribution
x_values=np.linspace(-5., 8., 100)

fig, ax=plt.subplots(3, sharex="col")
plt.title('Transform Gaussian')

ax[0].set_title('Standard Gaussian, (mu=0, sigma=1)')
ax[0].hist(x, bins=bins, density=True)
ax[0].plot(x_values, n_pdf(x_values), linestyle='dashed')

ax[1].set_title('f(x) = ax + b,  (a={}, b={})'.format(a, b))
ax[1].hist(f_1(x, a, b), bins=bins, density=True)
ax[1].plot(x_values, n_pdf(x_values, mu=b, sigma=a), linestyle='dashed')

ax[2].set_title('f(x) = x^2')
ax[2].hist(f_2(x), bins=bins, density=True)
ax[2].plot(x_values, pdf_2(x_values), linestyle='dashed')

plt.savefig(img_dir + "transformed_hist.png")
plt.show()
