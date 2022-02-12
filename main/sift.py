import warnings

warnings.filterwarnings("ignore")


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


# creating sift data
def image_to_sift(cv_img):
    sift = cv.SIFT_create()
    sift_array = []
    for img in cv_img:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        sift_array.append(des)
    return sift_array


# converting the SIFT data to feature
def create_sift_feature(sift_array):
    array = np.copy(sift_array)
    sift_feature = []
    for i in array:
        i_mean = np.mean(i)
        sift_feature.append(i_mean)
    return sift_feature


class SiftFeature:

    def getSiftFeature(self):
        img_list = read_image()
        sift_list = image_to_sift(img_list)
        sift_feature = create_sift_feature(sift_list)
        return sift_feature
