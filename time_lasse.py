import os
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_images(path, CAP=100):
    data = []
    files = os.listdir(path)
    c = 0

    for idx, file in enumerate(files):
        data.append(cv.imread(os.path.join(path, file)))
        c += 1
        if c >= CAP:
            break

    print("Data loaded successfully!")
    return np.array(data)


def speed_test(X, model):
    start = time.time()
    model.predict(X)
    return time.time() - start


def full_speed_test(X, model):
    times = []
    N = np.shape(X)[0]

    for i in range(N):
        input = np.expand_dims(X[i], 0)
        times.append(speed_test(input, model))

    return times

def show_estimations(y_est):
    for im in y_est:
        im = np.squeeze(im, axis=-1)
        plt.imshow(im, cmap='gray')
        plt.show()

############################################
# PARAMETERS
############################################

PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
DATA_PATH = PATH + "\\model2_time_benchmark\\images"
MODEL_PATH = PATH + "\\model2_time_benchmark\\model\\model2.h5"
model_name = "Model 2"
xlim = (30, 50)


# Begin script:

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Loading images...")
data = load_images(DATA_PATH)
d = data[0]
y_est = model.predict(np.expand_dims(d, axis=0))

show_estimations(y_est)

"""
print("Performing time benchmark...")
t_est = full_speed_test(data, model)

t_est = [t * 1000 for t in t_est]
N = len(t_est)
s2 = pd.DataFrame({'Time': t_est})
s1 = pd.DataFrame({'Model': [model_name] * N})

df = pd.concat([s1, s2], axis=1)

t_max = df[df["Model"] == model_name]["Time"].nlargest(2).to_numpy()
print("Slowest time: {} ms".format(t_max[0]))
print("Second slowest time: {} ms".format(t_max[1]))

sns.set_style("whitegrid")
g = sns.catplot(x="Time", y="Model", whis=[0, 95],
                height=3.5, aspect=1.5, palette="PuBuGn_d",
                kind="box", legend=False, data=df)

g.set_axis_labels("milliseconds", "")
g.set(xlim=xlim, yticklabels=model_name)
g.despine(trim=True)
plt.setp(g.ax.get_yticklabels(), rotation=30)
plt.show()
"""