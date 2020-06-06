import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
filenumbers = [i + 1 for i in range(5)]
data_files = ["unet{}_time.pickle".format(n) for n in filenumbers]
model_names = ["Model {}".format(n) for n in filenumbers]

data = []
pos = [1]

# Plot Details
outliers_props = dict(markerfacecolor='r', marker='X')

fig = plt.figure()
ax = fig.add_subplot(111)


dataframes = []

for idx, file in enumerate(data_files):
    d = load_pickle(PICKLE_PATH, file)
    N = len(d)
    s2 = pd.DataFrame({'Time': d})
    s1 = pd.DataFrame({'Model': [model_names[idx]] * N})

    dataframes.append(pd.concat([s1, s2], axis=1))

df = pd.concat(dataframes)

sns.set_style("whitegrid")
g = sns.catplot(x="Time", y="Model", whis=[0, 99],
                height=3.5, aspect=1.5, palette="PuBuGn_d",
                kind="box", legend=False, data=df)


g.set_axis_labels("milliseconds / image", "")
g.set(xlim=(4, 26), yticklabels=model_names)
g.despine(trim=True)
plt.setp(g.ax.get_yticklabels(), rotation=30)
plt.show()

