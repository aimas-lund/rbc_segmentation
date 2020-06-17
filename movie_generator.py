import os
import sys

import cv2 as cv


def key_extraction(filename):
    return int(filename.split('_')[1].split('.')[0])

DATADIR = "C:/Users/Aimas/Desktop/DTU/01-BSc/6_semester/01_Bachelor_Project/data/video/"

listdir = os.listdir(DATADIR)
shape = (640, 480)

num_files = len(listdir)
num_file = 0
out = cv.VideoWriter(os.path.join(DATADIR, 'project.avi'), cv.VideoWriter_fourcc(*'mp4v'), 30, (shape[0], shape[1]))

for img in sorted(listdir, key=key_extraction):
    image = cv.imread(os.path.join(DATADIR, img))
    out.write(image)
    num_file += 1
    sys.stdout.write("\r{} of {} files loaded".format(num_file, num_files))
    sys.stdout.flush()

out.release()
