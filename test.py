import pickle

import cv2

file_path = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\" \
            "data\\freja\\pickles\\0_20180613_3A_4mbar_2800fps_D1B.pickle"

file = open(file_path, 'rb')
X, y = pickle.load(file)
file.close()

cv2.imshow('original', X[2])
cv2.imshow('mask', y[2])
cv2.waitKey(0)
cv2.destroyAllWindows()