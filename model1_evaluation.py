import math

from evaluation import predict_sample
from unet_model import *

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = (HEIGHT, WIDTH, DEPTH)
NEW_SHAPE = (256, 256, 3)
BATCH_SIZE = 3
PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
#TRAINING_PATH = PATH + "\\data\\freja\\pickles"
#TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\aimas\\sample\\pickle"
TRAINING_FILE = "ph2_sample.pickle"
CALLBACK_PATH = PATH + "\\callbacks"
CALLBACK_NAME = "unet1-a.ckpt"
BIG_DATASET_PATH = PATH + "\\data\\freja\\pickles"
BIG_DATASET_NAME = "speed_test_data.pickle"
PICKLE_PATH = PATH + "\\pickle"
PICKLE_NAME = "unet1_time-a.pickle"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15
STRIDE = 1


#############################################
# Data Pre-Processing
#############################################

X_raw, y_raw = load_pickle(TRAINING_PATH, TRAINING_FILE)
X = []
y = []
"""
for i in range(len(X_raw)):
    img_x = tf.image.resize_with_pad(X_raw[i], 256, 256, method='bilinear')
    img_y = np.expand_dims(y_raw[i], -1)
    img_y = tf.image.resize_with_pad(img_y, 256, 256, method='bilinear')
    X.append(img_x.numpy() / 255.)
    y.append(img_y.numpy() / 255.)
"""

X = np.array(X_raw)
y = np.array(y_raw)
VALID_SIZE = math.floor(VALID_FRAC * len(X_raw))   # specifies the training data split

X_train = X[VALID_SIZE:]
X_valid = X[:VALID_SIZE]
y_train = y[VALID_SIZE:]
y_valid = y[:VALID_SIZE]

down_stack = [
    downsample(128, 3, strides=STRIDE),
    downsample(256, 3, strides=STRIDE)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3, strides=STRIDE),
    upsample(128, 3, strides=STRIDE)
]

# generate and compile model
model = unet_generator(SHAPE, down_stack, up_stack, strides=STRIDE)
model.load_weights(os.path.join(CALLBACK_PATH, CALLBACK_NAME))

y_est = predict_sample(X_valid, model)

EVAL_PATH = PATH + "\\pickle\\estimations"
save_pickle((y_valid, y_est), EVAL_PATH, "unet1-a_eval")
"""
big_X = load_big_data()
big_X_rescaled = []
print("Big Dataset loaded.")
reconfig_start = time.time()
for i in range(len(big_X)):
    img_x = tf.image.resize_with_pad(big_X[i], 256, 256, method='bilinear')
    big_X_rescaled.append(img_x.numpy() / 255.)

print(time.time() - reconfig_start)
print("Big Dataset reconfigured")
t = full_speed_test(big_X, model)
save_pickle(t, PICKLE_PATH, PICKLE_NAME)

#prec_rec_acc_plot(y_est, y_valid)
"""
