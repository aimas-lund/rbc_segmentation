import os
import pickle

import tensorflow as tf


def load_img(file, cast=None):
    """
    Loads an image and casts it as a tensorflow Tensor class
    :param file: Full path of image file
    :param cast: Type of cast performed on the image (default = no casting)
    :return: Tensor
    """
    file_type = file.split('.')[-1].lower()
    img = tf.io.read_file(file)

    if file_type == 'png':
        img = tf.image.decode_png(img)
    elif (file_type == 'jpg') or (file_type == 'jpeg'):
        img = tf.image.decode_jpeg(img)

    if cast is not None:
        img = tf.cast(img, cast)

    return img


def downsample(filters, size, apply_norm=True):
    """
    Downsamples an input using either Batchnorm, Dropout (optional) and ReLU
    All credit goes to https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/py
    :return: Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """
    Upsamples an input using either Batchnorm, Dropout (optional) and ReLU
    All credit goes to https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/py
    :return: Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_generator(shape, output_channels=1):
    input = tf.keras.layers.Input(shape=shape)
    model = input

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

    # define where in the model the skips should be made
    skips = []
    for down in down_stack:
        model = down(model)
        skips.append(model)

    skips = reversed(skips[:-1])

    # pair the upsampling (decode) with it's associated skip from the downsampling (encoding)
    for up, skip in zip(up_stack, skips):
        model = up(model)
        conc = tf.keras.layers.Concatenate()
        model = conc([model, skip])

    out_layer = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same',
        activation='softmax'
    )

    model = out_layer(model)
    model = tf.keras.Model(inputs=input, outputs=model)

    return model


def save_pickle(content, path, filename, override=False):
    """
    Saves a generated/trained model as a pickle file
    :param path: destination path
    :param filename: filename (without .pickle file extension)
    :param override: boolean to determine if already stored files may be overwritten
    :return:
    """
    f_path = os.path.join(path, filename) + '.pickle'
    i = 1
    rt_threshold = 50

    while (not override) and (os.path.exists(f_path)):  # while there is a file, add index until file doesn't exist
        new_filename = filename + ('_%i.pickle' % i)
        f_path = os.path.join(path, new_filename)
        i += 1

        if i > rt_threshold:
            raise RuntimeError("Files in directory exceeded %i files. Aborted!" % rt_threshold)

    # save model on specified path
    new_file = open(f_path, 'wb')
    pickle.dump(content, new_file)
    new_file.close()


def load_pickle(path, filename):
    f_path = os.path.join(path, filename + '.pickle')

    file = open(f_path, 'rb')
    output = pickle.load(file)
    file.close()

    return output


########################################
# Model creation
########################################


HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = [HEIGHT, WIDTH, DEPTH]
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\pickles"
TRAINING_FILE = "0_20180613_3A_4mbar_2800fps_D1B.pickle"
CALLBACK_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\callbacks"
CALLBACK_NAME = "unet1.ckpt"

OUTPUT_CHANNELS = 3

model = unet_generator(SHAPE)
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CALLBACK_PATH, CALLBACK_NAME),
                                                    save_weights_only=True,
                                                    verbose=1)
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# training the model
