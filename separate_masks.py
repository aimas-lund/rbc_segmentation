from preprocessing import mask_binarize
import os
import cv2
import matplotlib.pyplot as plt

# mask_separation("freja", 0)
file1 = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\annotations\\20180613_3A_4mbar_2800fps_D1_0907.mask.1.png"
file2 = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\data\\freja\\annotations\\20180613_3A_4mbar_2800fps_D1_0907.mask.2.png"
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)
b1, g1, r1 = cv2.split(img1)
b2, g2, r2 = cv2.split(img2)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret1, mask1 = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY)
ret2, mask2 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
mix = cv2.bitwise_or(mask1, mask2)
imgs = [mask1, mask2, mix]

for img in imgs:
    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()