import math

import cv2 as cv

from unet_model import *

SHAPE = (200, 800, 3)
#TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\pickles"
#TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\aimas\\sample\\pickles"
TRAINING_FILE = "ph2_sample.pickle"
CALLBACK_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\callbacks"
CALLBACK_NAME = "unet1-a.ckpt"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15
STRIDES = 2

#############################################
# Data Pre-Processing
#############################################

X_raw, y_raw = load_pickle(TRAINING_PATH, TRAINING_FILE)
X = []
y = []

for i in range(len(X_raw)):
    img_y = np.expand_dims(y_raw[i], -1)
    X.append(X_raw[i] / 255.)
    y.append(img_y / 255.)


X = np.array(X)
y = np.array(y)
VALID_SIZE = math.floor(VALID_FRAC * len(X_raw))   # specifies the training data split

X_train = X[VALID_SIZE:]
X_valid = X[:VALID_SIZE]
y_train = y[VALID_SIZE:]
y_valid = y[:VALID_SIZE]

# define encoding part of the model
down_stack = [
    downsample(128, 3, strides=STRIDES),
    downsample(256, 3, strides=STRIDES)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3, strides=STRIDES),
    upsample(128, 3, strides=STRIDES)
]

# generate and compile model
model = unet_generator(SHAPE, down_stack, up_stack, strides=STRIDES)
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CALLBACK_PATH, CALLBACK_NAME),
                                                    save_weights_only=True,
                                                    verbose=1)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)   # optimizer and specified learning rate
model.compile(optimizer=opt,
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

model.fit(X_train,
          y_train,
          epochs=25,
          batch_size=2,
          validation_data=(X_valid, y_valid),
          callbacks=[model_callback])  # Pass callback to training

cv.imshow('true', y_valid[3])
cv.waitKey(0)
cv.destroyAllWindows()
display_prediction(model.predict(X_valid[3]))