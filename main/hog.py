import warnings

warnings.filterwarnings("ignore")

from skimage.feature import hog
import glob
import cv2 as cv
import numpy as np


# read images from the folder
def read_image():
    # reading the image
    cv_img = []
    a = sorted(glob.glob("data/*"))
    for img in a:
        image = cv.imread(img)
        cv_img.append(image)
    return cv_img


# creating hog data
def image_to_hog(cv_img):
    fd_array = []
    for img in cv_img:
        fd, hog_image = hog(img, orientations=1, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True, transform_sqrt=True)
        fd_array.append(fd)
    return fd_array


# converting the hod data to feature
def create_hog_feature(fd_array):
    array = np.copy(fd_array)
    hog_feature = []
    for i in array:
        i_mean = np.mean(i)
        hog_feature.append(i_mean)
    return hog_feature


class HogFeature:

    def getHogFeature(self):
        img_list = read_image()
        hog_list = image_to_hog(img_list)
        hog_feature = create_hog_feature(hog_list)
        return hog_feature
