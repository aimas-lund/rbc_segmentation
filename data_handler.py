import glob
import os
import pickle
import random
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract(filename):
    return int(filename.split('_')[1].split('.')[0])

class DataHandler:
    def __init__(self, data_dir=None, categories=None, test=None):
        if test is None:
            self._TEST_CATEGORY = ['test']
        else:
            self._TEST_CATEGORY = test

        if categories is None:
            self.CATEGORIES = ["None", "Fetus", "Adult"]
        else:
            self.CATEGORIES = categories

        if data_dir is None:
            self.DATA_DIR = "C:/Users/Aimas/Desktop/DTU/01-BSc/6_semester/01_Bachelor_Project/data"
        else:
            self.DATA_DIR = data_dir

        self.IMG_RES = None
        self.training_data = []

    def detect_image_res(self):
        path = os.path.join(self.DATA_DIR, self.CATEGORIES[0])
        file = os.listdir(path)[0]
        image = cv2.imread(os.path.join(path, file))

        return np.shape(image)

    def fetch_images(self, grayscale=False, show=False):
        """
        Fetches data from the data directories.
        :param grayscale: boolean type to specify if images should be grayscale (normal is BRG blue-red-green).
        :param show: show the first sample of every category.
        :return: a numpy array containing all imported images.
        """
        images = []

        for category in self.CATEGORIES:
            path = os.path.join(self.DATA_DIR, category)  # define path to the different category data samples
            num_files = len(os.listdir(path))
            num_file = 0
            for img in os.listdir(path):
                try:
                    if grayscale:
                        images.append(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))
                    else:
                        images.append(cv2.imread(os.path.join(path, img)))
                    num_file += 1
                    sys.stdout.write("\r{} of {} files loaded".format(num_file, num_files))
                    sys.stdout.flush()
                except Exception:
                    print("Image {} in category {} skipped due to error!".format(img.index, category))
                    pass

            if show:
                if grayscale:
                    plt.imshow(images[0], cmap='gray')
                else:
                    plt.imshow(images[0])
                plt.show()

            return np.array(images)

    def create_training_data(self, grayscale=False, shuffle=True):
        """
        Creates training data for the attribute: self.training_data
        :param grayscale: (bool) indicating if images should be converted to gray scale
        :param shuffle: (bool) indicating if the method should return a shuffled sample of the data
        """
        self.training_data = []

        for category in self.CATEGORIES:
            path = os.path.join(self.DATA_DIR, category)
            class_num = self.CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    if grayscale:
                        image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    else:
                        image_array = cv2.imread(os.path.join(path, img))

                    self.training_data.append((image_array, class_num))

                except Exception:
                    print("Image {} in category {} skipped due to error!".format(img.index, category))
                    pass

        if shuffle:
            random.shuffle(self.training_data)  # the self.training_data will be shuffled, not just a sample

        sample_image = self.training_data[0][0]
        self.IMG_RES = sample_image.shape

    def shuffle_training_data(self):
        random.shuffle(self.training_data)

    def split_training_data(self, depth=1, reshape=False):
        """
        Splits the training data into features (X) and labels (y)
        :depth: (int) When reshaping the features, they have to be shaped according to what image type
        (default is 1, corresponding to e.g. grayscale images).
        :return: Feature * Label list
        """
        if len(self.training_data) < 1:
            raise ValueError
        else:
            features = []
            labels = []

            for sample in self.training_data:
                features.append(sample[0])
                labels.append(sample[1])

            if reshape:
                features = np.array(features).reshape(-1, self.IMG_RES[0], self.IMG_RES[1], depth)  # im not sure why
                # this step is necessary

            return np.array(features), np.array(labels)

    def save_training_data(self, filename='training_data'):
        """
        Saves the training data as a pickle file
        :param filename: (str) A specified filename to the ./pickle_data directory
        """
        if len(self.training_data) < 1:
            raise ValueError
        else:
            pkl_save = open("./pickle_data/" + filename + ".pickle", 'wb')
            pickle.dump(self.training_data, pkl_save)
            pkl_save.close()

    def load_training_data(self, filename='training_data'):
        """
        Loading a pickle file into the self.training_data attribute from the ./pickle_data directory
        :param filename: (str) A specified filename
        """
        try:
            pkl_load = open("./pickle_data/" + filename + ".pickle", 'rb')
            self.training_data = pickle.load(pkl_load)
            pkl_load.close()

        except FileNotFoundError:
            print("The specified filename does not exist in the pickle_data folder.")

    def png2jpg(self, input_dir, output_dir, filename='raw_', sort=True):

        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass

        img_list = os.listdir(input_dir)
        num_files = len(img_list)
        num_file = 0
        num_error = 0

        if sort:
            for img in sorted(img_list, key=extract):
                try:
                    png = cv2.imread(os.path.join(input_dir, img))
                    num_file += 1
                    if not cv2.imwrite(output_dir + filename + str(num_file - 1) + '.jpg', png):
                        num_error += 1
                    sys.stdout.write("\r{} of {} files converted from png to jpg - {} errors.".format(num_file,
                                                                                                      num_files,
                                                                                                      num_error))
                    sys.stdout.flush()
                except Exception:
                    print("Image {} skipped due to error!".format(img.index))
        else:
            for img in img_list:
                try:
                    png = cv2.imread(os.path.join(input_dir, img))
                    num_file += 1
                    if not cv2.imwrite(output_dir + filename + str(num_file - 1) + '.jpg', png):
                        num_error += 1
                    sys.stdout.write("\r{} of {} files converted from png to jpg - {} errors.".format(num_file,
                                                                                                      num_files,
                                                                                                      num_error))
                    sys.stdout.flush()
                except Exception:
                    print("Image {} skipped due to error!".format(img.index))


    def sample_images(source_path, dest_path, samples=100):

        source_path = source_path + "/*.png"
        to_be_moved = random.sample(glob.glob(source_path), samples)

        for f in enumerate(to_be_moved, 1):
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.copy(f[1], dest_path)


    def prepare_data(self, index, type='freja', from_pickle=True, save=False, gray=False):
        if type == 'freja':
            # Fetches sample from frejas data
            path = os.path.join(self.DATA_DIR, type + "/samples/png")
            folder = os.listdir(path)[index]
            path = os.path.join(path, folder)

        for img in os.listdir(path):
            if gray:
                image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            else:
                image_array = cv2.imread(os.path.join(path, img))



