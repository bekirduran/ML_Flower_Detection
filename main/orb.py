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


# creating orb data
def image_to_orb(cv_img):
    orb = cv.ORB_create()
    orb_array = []
    for img in cv_img:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        orb_array.append(des)
    return orb_array


# converting the orb data to feature
def create_orb_feature(fd_array):
    array = np.copy(fd_array)
    orb_feature = []
    for i in array:
        i_mean = np.mean(i)
        orb_feature.append(i_mean)
    return orb_feature


class OrbFeature:

    def getOrbFeature(self):
        img_list = read_image()
        orb_list = image_to_orb(img_list)
        orb_feature = create_orb_feature(orb_list)
        return  orb_feature





