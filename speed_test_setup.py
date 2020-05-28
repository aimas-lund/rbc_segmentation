import os

import cv2 as cv
import numpy as np

from unet_model import save_pickle

PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
DATA_PATH = PATH + "\\data\\freja\\all\\20180613_3A_4mbar_2800fps_D2B"
PICKLE_PATH = PATH + "\\data\\freja\\pickles"
PICKLE_NAME = "speed_test_data.pickle"
SAMPLE_SIZE = 25000

data = []
files = os.listdir(DATA_PATH)
N_files = len(files)

for idx, file in enumerate(files):
    print("Image {} of {}".format(idx + 1, SAMPLE_SIZE))
    data.append(cv.imread(os.path.join(DATA_PATH, file)))

    if (idx + 1) >= SAMPLE_SIZE:
        break

data = np.array(data)
save_pickle(data, PICKLE_PATH, PICKLE_NAME)