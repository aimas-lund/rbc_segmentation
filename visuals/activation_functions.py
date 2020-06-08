from copy import copy
from math import exp, tanh

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1 + exp(-x))

def relu(x):
    return max(0, x)

x = np.arange(start=-5, stop=5, step=0.05)
y_map = [
    sigmoid,
    tanh,
    relu]

titles = [
    "Sigmoid",
    "Tanh",
    "ReLU"]

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

for idx, ax in enumerate(axs):
    y = copy(x)

    for i in range(len(x)):
        y[i] = y_map[idx](x[i])

    ax.plot(x, y, c='#000000')
    ax.set_title(titles[idx])
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.show()