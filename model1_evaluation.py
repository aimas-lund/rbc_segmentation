import math

import numpy as np

from evaluation import predict_sample, TPR_FPR_plot, prec_rec_acc_plot
from unet_model import *

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = (HEIGHT, WIDTH, DEPTH)
NEW_SHAPE = (256, 256, 3)
BATCH_SIZE = 3
PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
TRAINING_PATH = PATH + "\\data\\freja\\pickles"
TRAINING_PATH_X = PATH + "\\data\\freja\\samples\\png\\0_20180613_3A_4mbar_2800fps_D1B"
TRAINING_PATH_Y = PATH + "\\data\\freja\\annotations_combined\\0_20180613_3A_4mbar_2800fps_D1B"
CALLBACK_PATH = PATH + "\\callbacks"
CALLBACK_NAME = "unet1.ckpt"
TRAINED_MODEL_PATH = PATH + "\\trained_models\\unet1"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15


#############################################
# Data Pre-Processing
#############################################

X_raw, y_raw = load_pickle(TRAINING_PATH, TRAINING_FILE)
X = []
y = []

for i in range(len(X_raw)):
    img_x = tf.image.resize_with_pad(X_raw[i], 256, 256, method='bilinear')
    img_y = np.expand_dims(y_raw[i], -1)
    img_y = tf.image.resize_with_pad(img_y, 256, 256, method='bilinear')
    X.append(img_x.numpy() / 255.)
    y.append(img_y.numpy() / 255.)


X = np.array(X)
y = np.array(y)
VALID_SIZE = math.floor(VALID_FRAC * len(X_raw))   # specifies the training data split

X_train = X[VALID_SIZE:]
X_valid = X[:VALID_SIZE]
y_train = y[VALID_SIZE:]
y_valid = y[:VALID_SIZE]

down_stack = [
    downsample(128, 3),
    downsample(256, 3)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3),
    upsample(128, 3)
]

# generate and compile model
model = unet_generator(NEW_SHAPE, down_stack, up_stack)
model.load_weights(os.path.join(CALLBACK_PATH, CALLBACK_NAME))

y_est = predict_sample(X_valid, model)

TPR_FPR_plot(y_est, y_valid)
prec_rec_acc_plot(y_est, y_valid)

