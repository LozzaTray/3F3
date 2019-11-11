import numpy as np
from pdfs import n_pdf

def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""

    def ksd(x_axis):

        prob = [n_pdf(x_i, data, width) for x_i in x_axis]
        pdf = [np.average(pr) for pr in prob]  # each row is one x value
        return np.array(pdf)
    return ksd