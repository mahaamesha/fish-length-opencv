import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image, destroy_all=True):
    cv.imshow(title, image)
    cv.waitKey(0)
    if destroy_all:
        cv.destroyAllWindows()

def process_color(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # define range of color in HSV
    lower = np.array([110,50,50])
    upper = np.array([130,255,255])
    
    # Threshold the HSV image to get only in range colors
    mask = cv.inRange(hsv, lower, upper)
    
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(image,image, mask= mask)