from unet_model import SegmentationModel

seg_model = SegmentationModel()
seg_model.model = 1

path = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project\\trained_models"
filename = "test"
seg_model.save_model(path, filename)
seg_model.save_model(path, filename)
