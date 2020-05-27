import math

from unet_model import *

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = (HEIGHT, WIDTH, DEPTH)
NEW_SHAPE = (256, 256, 3)
BATCH_SIZE = 3
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\data\\freja\\pickles"
TRAINING_PATH_X = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja" \
                  "\\samples\\png\\0_20180613_3A_4mbar_2800fps_D1B"
TRAINING_PATH_Y = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja" \
                  "\\annotations_combined\\0_20180613_3A_4mbar_2800fps_D1B"
CALLBACK_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\callbacks"
CALLBACK_NAME = "unet3.ckpt"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
D_TYPE = tf.float32
OUTPUT_CHANNELS = 1
VALID_FRAC = 0.15
DENSE_LAYERS = 4
NEURON_NUM = 400

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
    y.append(img_y.numpy().flatten() / 255.)


X = np.array(X)
y = np.array(y)
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
    downsample(32, 3),
    downsample(64, 3),
    downsample(128, 3),
    downsample(256, 3)
]

# define the decoding part of the model
up_stack = [
    upsample(256, 3),
    upsample(128, 3),
    upsample(64, 3),
    upsample(32, 3)
]

# generate and compile model
model = unet_dense_generator(NEW_SHAPE, down_stack, up_stack,
                            dense_layers=DENSE_LAYERS,
                            neuron_num=NEURON_NUM)
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
          epochs=15,
          batch_size=1,
          validation_data=(X_valid, y_valid),
          callbacks=[model_callback])  # Pass callback to training

pred = model.predict(np.expand_dims(X_valid[8], axis=0))

y_v = np.reshape(y_valid[8], (256, 256, 1))
pred = np.reshape(pred, (256, 256, 1))

display_prediction(X_valid[8])
display_prediction(y_v)
display_prediction(pred)
