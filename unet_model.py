import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


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


def downsample(filters, size, apply_norm=True, strides=2):
    """
    Downsamples an input using either Batchnorm, Dropout (optional) and ReLU
    All credit goes to https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/py
    :return: Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        Conv2D(filters, size,
               strides=strides,
               padding='same',
               kernel_initializer=initializer,
               use_bias=True))

    if apply_norm:
        result.add(BatchNormalization())

    result.add(ReLU())

    return result


def upsample(filters, size, apply_dropout=False, strides=2):
    """
    Upsamples an input using either Batchnorm, Dropout (optional) and ReLU
    All credit goes to https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/py
    :return: Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        Conv2DTranspose(filters, size,
                        strides=strides,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result


def unet_generator(shape, down_stack, up_stack,
                   output_channels=1, strides=2):
    input = Input(shape=shape)
    model = input

    # define where in the model the skips should be made
    skips = []
    for down in down_stack:
        model = down(model)
        skips.append(model)

    skips = reversed(skips[:-1])

    # pair the upsampling (decode) with it's associated skip from the downsampling (encoding)
    for up, skip in zip(up_stack, skips):
        model = up(model)
        conc = Concatenate()
        model = conc([model, skip])

    out_layer = Conv2DTranspose(
        output_channels,
        kernel_size=3,
        strides=strides,
        padding='same',
        activation='sigmoid'
    )

    model = out_layer(model)
    model = tf.keras.Model(inputs=input, outputs=model)

    return model


def unet_dense_generator(shape, down_stack, up_stack,
                         output_channels=1,
                         dense_layers=2,
                         neuron_num=865,
                         strides=2):
    input = Input(shape=shape)
    model = input

    # define where in the model the skips should be made
    skips = []
    for down in down_stack:
        model = down(model)
        skips.append(model)

    skips = reversed(skips[:-1])

    # pair the upsampling (decode) with it's associated skip from the downsampling (encoding)
    for up, skip in zip(up_stack, skips):
        model = up(model)
        conc = Concatenate()
        model = conc([model, skip])

    transpose = Conv2DTranspose(
        output_channels,
        kernel_size=3,
        strides=strides,
        padding='same',
        activation='relu'
    )

    model = transpose(model)
    model = Flatten()(model)

    # add dense layers
    output_neurons = shape[0] * shape[1]

    for _ in range(dense_layers - 1):
        dense = Dense(neuron_num,
                      activation='relu')
        model = dense(model)

    output_layer = Dense(output_neurons,
                         activation='sigmoid')

    model = output_layer(model)
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
    filetype = filename.split('.')[-1]

    if filetype.lower() != 'pickle':
        f_path = os.path.join(path, filename + '.pickle')
    else:
        f_path = os.path.join(path, filename)
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
    filetype = filename.split('.')[-1]

    if filetype.lower() != 'pickle':
        f_path = os.path.join(path, filename + '.pickle')
    else:
        f_path = os.path.join(path, filename)

    file = open(f_path, 'rb')
    output = pickle.load(file)
    file.close()

    return output


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display_prediction(tensor):
    """
    Convert tensor to a plottable format
    :param tensor:
    :return:
    """
    title = "Output Image"
    img = tensor
    if len(np.shape(tensor)) > 3:
        img = img[0]

    plt.title(title)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
    plt.axis('off')
    plt.show()


def rescale_images(X_raw, y_raw, size=(256, 256), flatten=False):
    X = []
    y = []

    for i in range(len(X_raw)):
        img_x = tf.image.resize_with_pad(X_raw[i], size[0], size[1], method='bilinear')
        img_y = np.expand_dims(y_raw[i], -1)
        img_y = tf.image.resize_with_pad(img_y, size[0], size[1], method='bilinear')
        X.append(img_x.numpy() / 255.)
        if flatten:
            y.append(img_y.numpy().flatten() / 255.)
        else:
            y.append(img_y.numpy() / 255.)

    X = np.array(X)
    y = np.array(y)
    print("Rescale Complete")

    return X, y


def rescale_X(X_raw, size=(256, 256)):
    X = []

    for i in range(len(X_raw)):
        img_x = tf.image.resize_with_pad(X_raw[i], size[0], size[1], method='bilinear')
        X.append(img_x.numpy() / 255.)

    X = np.array(X)
    print("Rescale Complete")

    return X
