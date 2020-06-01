import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, norm

from unet_model import load_pickle


def get_percentiles(x, lower=10., upper=90.):
    return np.percentile(np.array(x), lower), np.percentile(np.array(x), upper)


def calc_distribution(y, type='norm', lower=0.01, upper=99.99, points=100):
    lo, up = get_percentiles(y, lower, upper)
    X = np.linspace(lo, up, points)

    if type == 'norm':
        p1, p2 = norm.fit(y)
        Y = norm.pdf(X, p1, p2)

        return X, Y

    elif type == 'skewed':
        p1, p2, p3 = skewnorm.fit(y)
        Y = skewnorm.pdf(X, p1, p2, p3)

        return X, Y

    else:
        raise AttributeError("'type' not recognized.")


PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
PICKLE_PATH = PATH + "\\pickle"

data_files = os.listdir(PICKLE_PATH)
data = []
pos = [1]

# Plot Details
outliers_props = dict(markerfacecolor='r', marker='X')

fig = plt.figure()
ax = fig.add_subplot(111)

for file in data_files:
    d = load_pickle(PICKLE_PATH, file)
    x, y = calc_distribution(d)

    ax.boxplot(d, positions=pos, widths=0.6, showmeans=True,
                flierprops=outliers_props)
    pos = [p + 1 for p in pos]

ax.set_xlabel('Model Number')
ax.set_ylabel('ms / image')
ax.grid(linestyle='-')
plt.show()

