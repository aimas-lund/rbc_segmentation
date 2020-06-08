import math

from evaluation import *
from speed_test_setup import *
from unet_model import *

SHAPE = (200, 800, 3)
NEW_SHAPE = (128, 512, 3)
BATCH_SIZE = 1
PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
#TRAINING_PATH = PATH + "\\data\\freja\\pickles"
#TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
TRAINING_PATH = PATH + "\\data\\aimas\\sample\\pickles"
TRAINING_FILE = "ph2_sample.pickle"
CALLBACK_PATH = PATH + "\\callbacks"
CALLBACK_NAME = "unet2-a.ckpt"
PICKLE_PATH = PATH + "\\pickle\\time"
PICKLE_NAME = "unet2-a_time.pickle"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15
STRIDES = 2


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
    downsample(32, 3, strides=STRIDES),
    downsample(64, 3, strides=STRIDES),
    downsample(128, 3, strides=STRIDES),
    downsample(256, 3, strides=STRIDES)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3, strides=STRIDES),
    upsample(128, 3, strides=STRIDES),
    upsample(64, 3, strides=STRIDES),
    upsample(32, 3, strides=STRIDES)
]

# generate and compile model
model = unet_generator(NEW_SHAPE, down_stack, up_stack, strides=STRIDES)
model.load_weights(os.path.join(CALLBACK_PATH, CALLBACK_NAME))

"""
y_est = predict_sample(X_valid, model)
EVAL_PATH = PATH + "\\pickle\\estimations"
save_pickle((y_valid, y_est), EVAL_PATH, "unet2-a_eval")

"""
big_X = load_big_data(type='aimas')
big_X_rescaled = []
t_transform = []
print("Big Dataset loaded.")

for i in range(len(big_X)):
    reconfig_start = time.time()
    img_x = tf.image.resize_with_pad(big_X[i], NEW_SHAPE[0],
                                     NEW_SHAPE[1], method='bilinear')
    t_transform.append(reconfig_start - time.time())
    big_X_rescaled.append(img_x.numpy() / 255.)

print("Big Dataset reconfigured")

#t = speed_test(X, model)
#save_pickle(t, PICKLE_PATH, PICKLE_NAME)
t_est = full_speed_test(big_X_rescaled, model)
save_pickle((t_transform, t_est), PICKLE_PATH, PICKLE_NAME)


# show_estimations(y_est)
# TPR_FPR_plot(y_est, y_valid)
# prec_rec_acc_plot(y_est, y_valid)

