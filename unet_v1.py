import math

from unet_model import *

SHAPE = (120, 260, 3)
NEW_SHAPE = (256, 256, 3)
BATCH_SIZE = 3
PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
#TRAINING_PATH = PATH + "\\data\\aimas\\sample\\pickles"
#TRAINING_FILE = "ph2_sample.pickle"
TRAINING_PATH = PATH + "\\data\\freja\\pickles"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
LOG_PATH = PATH + "\\logs"
CALLBACK_PATH = PATH + "\\callbacks"
CALLBACK_NAME = "unet1.ckpt"
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
model = unet_generator(NEW_SHAPE, down_stack, up_stack, strides=STRIDES)
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
          epochs=10,
          batch_size=1,
          validation_data=(X_valid, y_valid),
          callbacks=[model_callback])  # Pass callback to training
