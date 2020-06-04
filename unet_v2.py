import math

from unet_model import *

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = (HEIGHT, WIDTH, DEPTH)
NEW_SHAPE = (128, 512, 3)
BATCH_SIZE = 3
PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
TRAINING_PATH = PATH + "\\data\\aimas\\sample\\pickles"
TRAINING_PATH_X = PATH + "\\data\\freja\\samples\\png\\0_20180613_3A_4mbar_2800fps_D1B"
TRAINING_PATH_Y = PATH + "\\data\\freja\\annotations_combined\\0_20180613_3A_4mbar_2800fps_D1B"
CALLBACK_PATH = PATH + "\\callbacks"
CALLBACK_NAME = "unet2-a.ckpt"
TRAINING_FILE = "ph2_sample.pickle"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15

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
    downsample(32, 3, strides=1),
    downsample(64, 3, strides=1),
    downsample(128, 3, strides=1),
    downsample(256, 3, strides=1)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3, strides=1),
    upsample(128, 3, strides=1),
    upsample(64, 3, strides=1),
    upsample(32, 3, strides=1)
]

# generate and compile model
model = unet_generator(NEW_SHAPE, down_stack, up_stack, strides=1)
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CALLBACK_PATH, CALLBACK_NAME),
                                                    save_weights_only=True,
                                                    verbose=1)
opt = tf.keras.optimizers.Adam(learning_rate=0.00005)   # optimizer and specified learning rate
model.compile(optimizer=opt,
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()   # prints a summary of the model
tf.keras.utils.plot_model(model, show_shapes=True)


#############################################
# Model Fitting
#############################################

# training the model
model.fit(X_train,
          y_train,
          epochs=10,
          batch_size=1,
          validation_data=(X_valid, y_valid),
          callbacks=[model_callback])  # Pass callback to training

display_prediction(model.predict(np.expand_dims(X_valid[3], 0)))
display_prediction(model.predict(np.expand_dims(X_valid[4], 0)))
display_prediction(model.predict(np.expand_dims(X_valid[5], 0)))
display_prediction(model.predict(np.expand_dims(X_valid[6], 0)))