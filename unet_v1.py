import math

from preprocessing import preprocess_image
from unet_model import *

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = (HEIGHT, WIDTH, DEPTH)
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\pickles"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
CALLBACK_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\callbacks"
CALLBACK_NAME = "unet1.ckpt"
D_TYPE = tf.float32

OUTPUT_CHANNELS = 1
VALID_FRAC = 0.2

# define encoding part of the model
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
model = unet_generator(SHAPE, down_stack, up_stack)
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CALLBACK_PATH, CALLBACK_NAME),
                                                    save_weights_only=True,
                                                    verbose=1)
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# training the model
X_raw, y_raw = load_pickle(TRAINING_PATH, TRAINING_FILE)
X = []
y = []

for i in range(len(X_raw)):
    X.append(preprocess_image(X_raw[i], D_TYPE))
    y.append(preprocess_image(y_raw[i], D_TYPE))


X = np.array(X)
y = np.array(y)
y = np.expand_dims(y, axis=-1) # add an extra dimension to the true data

VALID_SIZE = math.floor(VALID_FRAC * len(X_raw))

X_train = X[VALID_SIZE:]
X_valid = X[:VALID_SIZE]
y_train = y[VALID_SIZE:]
y_valid = y[:VALID_SIZE]

#dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#dataset_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

print("\nData Preprocessing Complete!")

model.fit(X_train,
          y_train,
          epochs=3,
          batch_size=2,
          validation_data=(X_valid, y_valid),
          callbacks=[model_callback])  # Pass callback to training

pred = model.predict(np.expand_dims(X_valid[8], axis=0))
pred_mask = create_mask(pred)
pred_img = tf.keras.preprocessing.image.array_to_img(pred_mask)
display_prediction(X_train[5])
display_prediction(y_train[5])
display_prediction(pred_mask)
display_prediction(pred)