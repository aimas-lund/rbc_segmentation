import os

import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

path = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\all\\" \
       "20180613_3A_4mbar_2800fps_D1B\\20180613_3A_4mbar_2800fps_D1_0156.png"
path1 = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\" \
        "data\\aimas\\01-4-40x-6mbar-500fps\\raw\\raw_226.png"
dest = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\graphics"
paths = [path, path1]
name = "laplacian"
ori_names = ["Freja", "Aimas"]

kernel_size = 3
ddepth = cv2.CV_32F

for idx, val in enumerate(paths):
    ori_img = cv2.imread(val)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    if idx == 1:
        cv2.imshow('no gamma', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        gamma = adjust_gamma(img, 1.5)
        cv2.imshow('w. gamma', gamma)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = cv2.GaussianBlur(img,(3,3),0)
    img_laplace = cv2.Laplacian(img, ddepth, ksize=kernel_size)
    filename = name + '_%i' % idx + '.png'
    ori_filename = 'original_' + ori_names[idx] + '_%i' % idx + '.png'

    cv2.imwrite(os.path.join(dest, filename), img_laplace)
    cv2.imwrite(os.path.join(dest, ori_filename), ori_img)