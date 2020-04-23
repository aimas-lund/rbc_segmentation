import os
import pickle

import tensorflow as tf

HEIGHT = 120
WIDTH = 260
DEPTH = 3
SHAPE = [HEIGHT, WIDTH, DEPTH]
TRAINING_PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\pickles"

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
    elif (file_type == 'jpg') or (file_type == 'jpeg') :
        img = tf.image.decode_jpeg()

    if cast is not None:
        img = tf.cast(img, cast)

    return img


def downsample(filters, size, apply_norm=True):
    """
        Downsamples an input using either Batchnorm, Dropout (optional) and ReLU
        All credit goes to https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
        :return: Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())  #TODO: potentially also implement LeakyReLU

    return result


def upsample(filters, size, apply_dropout=False):
    """
    Upsamples an input using either Batchnorm, Dropout (optional) and ReLU
    All credit goes to https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    :return: Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_generator(output_chan, shape, output_channels=1):
    input = tf.keras.layers.Input(shape=shape)
    model = input

    """
    Intermediate layers go here
    """

    out_layer = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same',
        activation='sigmoid'
    )

    model = out_layer(model)

    return tf.keras.Model(inputs=input, outputs=model)

def save_model(self, model, path, filename, override=False):
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

    while (not override) and (os.path.exists(f_path)):   # while there is a file, add index until file doesn't exist
        new_filename = filename + ('_%i.pickle' % i)
        f_path = os.path.join(path, new_filename)
        i += 1

        if i > rt_threshold:
            raise RuntimeError("Files in directory exceeded %i files. Aborted!" % rt_threshold)

    # save model on specified path
    new_file = open(f_path, 'wb')
    pickle.dump(model, new_file)
    new_file.close()

