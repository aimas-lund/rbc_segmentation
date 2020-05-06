from unet_model import *

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = [HEIGHT, WIDTH, DEPTH]
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\pickles"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
CALLBACK_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\callbacks"
CALLBACK_NAME = "unet1.ckpt"

OUTPUT_CHANNELS = 3

# define encoding part of the model
down_stack = [
    downsample(64, 3),
    downsample(128, 3),
]

# define the decoding part of the model
up_stack = [
    upsample(128, 3),
    upsample(64, 3),
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

# training the model
