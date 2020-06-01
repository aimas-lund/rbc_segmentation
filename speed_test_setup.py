import os

import cv2 as cv
import numpy as np


def load_big_data():

    PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
    DATA_PATH = PATH + "\\data\\freja\\all\\20180613_3A_4mbar_2800fps_D2B"
    SAMPLE_SIZE = 1000

    data = []
    files = os.listdir(DATA_PATH)

    for idx, file in enumerate(files):
        print("Image {} of {}".format(idx + 1, SAMPLE_SIZE))
        data.append(cv.imread(os.path.join(DATA_PATH, file)))

        if (idx + 1) >= SAMPLE_SIZE:
            break

    return np.array(data)