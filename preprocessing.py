# coding: utf-8

import os
import pickle

import cv2
import numpy as np

DATA_DIR = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data"
dest = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\" \
       "freja\\annotations_combined\\0_20180613_3A_4mbar_2800fps_D1B"
source = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\" \
         "data\\freja\\annotations\\0_20180613_3A_4mbar_2800fps_D1B"
pckl = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\" \
         "data\\freja\\pickles"
sample = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\" \
         "data\\freja\\samples\\png\\0_20180613_3A_4mbar_2800fps_D1B"
y_path = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\" \
         "data\\freja\\annotations_combined\\0_20180613_3A_4mbar_2800fps_D1B"

def sort_order(filename):
    return int(filename.split('_')[1].split('.')[0])

def find_masks(filenames, token='mask', bool=False):
    if type(filenames) is list:
        filenames = np.array(filenames)

    masks_b = np.full(len(filenames), False)   # allocate memory

    for idx, file in enumerate(filenames):
        if token in file:
            masks_b[idx] = True

    if bool:
        return masks_b
    else:
        return filenames[masks_b]

def mask_generate_blank(path, dest):
    files = os.listdir(path)

    for file in files:
        if not 'mask' in file:
            filename = file.split('.')[0]
            data = cv2.imread(os.path.join(path, file))
            h, w, _ = data.shape
            img = np.zeros((h, w), dtype=np.uint8)

            cv2.imwrite(os.path.join(dest, filename) + '_mask.png', img)


def mask_separation(extractor, index):
    upper_path = os.path.join(DATA_DIR, extractor.lower())
    dest_path = os.path.join(upper_path, "annotations")
    smpl_path = os.path.join(upper_path, "samples/png/")
    folder = os.listdir(smpl_path)[index]
    smpl_path = os.path.join(smpl_path, folder)

    if extractor == 'freja':
        files = os.listdir(smpl_path)

        masks = find_masks(files)

        for mask in masks:
            img = cv2.imread(os.path.join(smpl_path, mask))
            cv2.imwrite(os.path.join(dest_path, mask), img)

def masks_combine(path, include_trunc=True, dest=None):

    # match separate masks belonging to the same image
    files = os.listdir(path)
    mapping = dict()

    for file in files:
        # split the filename such that we we can match the masks
        identifier = file.split('.')[0]

        # add files to a dictionary mapping a file-id to its masks
        if identifier in mapping:
            mapping[identifier].append(file)
        else:
            mapping[identifier] = [file]

    # define destination path for combined binaries
    if dest is None:
        destination = os.path.join(path, "combined")
    else:
        if not os.path.exists(dest):
            os.mkdir(dest)

        destination = dest

    if not os.path.exists(destination):
        os.mkdir(destination)

    # combine binaries
    for key, masks in mapping.items():
        binaries = []

        for mask in masks:
            img = cv2.imread(os.path.join(path, mask))
            _, green, red = cv2.split(img)
            height, width, _ = img.shape

            # convert green and red part of images to binary
            if red.max() > 0:
                _, binary = cv2.threshold(red, 1, 255, cv2.THRESH_BINARY)
            elif (include_trunc) and (green.max() > 0):
                _, binary = cv2.threshold(green, 1, 255, cv2.THRESH_BINARY)
            else:
                binary = np.zeros((height, width), np.uint8)    # make a blank image

            binaries.append(binary)

        combined = binaries[0]
        if len(binaries) > 1:
            for i in range(1, len(binaries)):
                combined = cv2.bitwise_or(combined, binaries[i])    # Combine all associated masks bitwise
        filename = os.path.join(destination, (key + "_mask.png"))
        cv2.imwrite(filename, combined)

    return None


def extract_sample(path):
    files = os.listdir(path)
    indices = []
    imgs = []

    for file in files:
        if not 'mask' in file:
            filepath = os.path.join(path, file)
            imgs.append(cv2.imread(filepath))

            # extract index
            lhs = file.split('.')[0]
            indices.append(int(lhs.split('_')[-1]))

    return imgs, indices


def pickle_training_data(X_path, y_path, dest, filename='training_data'):
    # Fetching X data
    X, keys = extract_sample(X_path)

    # Fetching y data
    y_files = os.listdir(y_path)
    y = []
    y_indices = []


    for file in y_files:
        y.append(cv2.imread(os.path.join(y_path, file), 0))
        y_indices.append(int(file.split('_')[-2]))

    #X = np.sort(X)
    #y = np.sort(y)
    data = (X, y)

    if not os.path.exists(dest):
        os.mkdir(dest)

    pckl = open(os.path.join(dest, filename) + '.pickle', 'wb')
    pickle.dump(data, pckl)
    pckl.close()

    print("Pickle saved to:\n" + dest)



mask_generate_blank(sample, dest)
masks_combine(source, dest=dest)
pickle_training_data(sample, y_path, pckl, filename="0_20180613_3A_4mbar_2800fps_D1B")

