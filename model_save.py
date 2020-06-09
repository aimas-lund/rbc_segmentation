import math

import cv2 as cv

from evaluation import *
from speed_test_setup import *
from unet_model import *

SHAPE = (120, 260, 3)
NEW_SHAPE = (256, 256, 3)
BATCH_SIZE = 1
NAME = "unet2"
PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
TRAINING_PATH = PATH + "\\data\\freja\\pickles"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
#TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\aimas\\sample\\pickles"
#TRAINING_FILE = "ph2_sample.pickle"
CALLBACK_PATH = PATH + "\\callbacks"
CALLBACK_NAME = NAME + ".ckpt"
BIG_DATASET_PATH = PATH + "\\data\\freja\\pickles"
BIG_DATASET_NAME = "speed_test_data.pickle"
PICKLE_PATH = PATH + "\\pickle\\time"
PICKLE_NAME = NAME + "_time.pickle"
METRIC_PATH = PATH + "\\pickle\\metrics"
METRIC_NAME = NAME + "_metrics.pickle"
SAVED_MODEL_PATH = PATH + "\\models"
MODEL_NAME = "model2.h5"
TRANSFORMED_PATH = PATH + "\\zip"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15
STRIDE = 2


#############################################
# Data Pre-Processing
#############################################

X_raw, y_raw = load_pickle(TRAINING_PATH, TRAINING_FILE)

X, y = rescale_images(X_raw, y_raw, size=NEW_SHAPE)

VALID_SIZE = math.floor(VALID_FRAC * len(X_raw))   # specifies the training data split

X_train = X[VALID_SIZE:]
X_valid = X[:VALID_SIZE]
y_train = y[VALID_SIZE:]
y_valid = y[:VALID_SIZE]


#############################################
# Model Generation
#############################################

# define encoding part of the model
down_stack = [
    downsample(32, 3, strides=STRIDE),
    downsample(64, 3, strides=STRIDE),
    downsample(128, 3, strides=STRIDE),
    downsample(256, 3, strides=STRIDE)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3, strides=STRIDE),
    upsample(128, 3, strides=STRIDE),
    upsample(64, 3, strides=STRIDE),
    upsample(32, 3, strides=STRIDE)
]

# generate and compile model
model = unet_generator(NEW_SHAPE, down_stack, up_stack, strides=STRIDE)
model.load_weights(os.path.join(CALLBACK_PATH, CALLBACK_NAME))

big_X = load_big_data(type='freja')
t_transform = []
print("Big Dataset loaded.")

for i in range(len(big_X)):
    reconfig_start = time.time()
    img_x = tf.image.resize_with_pad(big_X[i], NEW_SHAPE[0],
                                     NEW_SHAPE[1], method='bilinear')
    t_transform.append(reconfig_start - time.time())
    img = img_x.numpy()
    cv.imwrite(TRANSFORMED_PATH + "\\img%i.png" % i, img)