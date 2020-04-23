import os
import pickle

class SegmentationModel:
    def __init__(self, model_type='unet'):
        self.type = model_type
        self.model = None
        self.pickle_path = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\trained_models"
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def save_model(self, path, filename, override=False):
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

        new_file = open(f_path, 'wb')
        pickle.dump(self.model, new_file)
        new_file.close()