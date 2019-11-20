import numpy as np
import matplotlib.pyplot as plt


def p(y):
    "PDF for exponential"
    return np.exp(-y)


def f(x):
    """inverse CDF for exponential pdf"""
    return - np.log(1 - x)

bins = 30
N = 1000
x = np.random.rand(N)
y_vals = np.linspace(0, 5., 100)

y = f(x)

fig = plt.figure()

plt.title('Inverse cdf method for exponential (N={})'.format(N))
plt.hist(y, bins=bins, density=True, label="normalised frequency")
plt.plot(y_vals, p(y_vals), linestyle='dashed', label="desired pdf")
plt.xlabel('y')
plt.ylabel('Normalised frequency')
plt.legend()

img_dir = "/mnt/c/Users/ltray/Documents/Cambridge/3F3/img/"
plt.savefig(img_dir + "exponential.png")
plt.show()