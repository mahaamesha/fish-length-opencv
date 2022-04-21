from cv2 import ellipse
from matplotlib import markers
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import numpy as np
import imutils
import cv2 as cv

import src.mafunction as f

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cvt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

f.show_image("ORI", image, False)
f.show_image("HSV", cvt, False)

imgaus = cv.GaussianBlur(cvt, (51, 51), 0)
f.show_image("gaus", imgaus, False)

ret, thresh = cv.threshold(imgaus,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
f.show_image("thrs", thresh, False)

kernel = np.ones( (3,3), np.uint8 )
erodila = cv.erode(thresh, kernel, iterations=3)
erodila = cv.dilate(erodila, kernel, iterations=1)
f.show_image("erodila", erodila, False)

res = cv.bitwise_and(image,image, mask=erodila)
f.show_image("res", res, False)

edged = cv.Canny(erodila, 200, 255)
f.show_image("Edged", edged, False)

# Find many object there
cnts = cv.findContours(erodila, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("Total number of contours: ", len(cnts))

f.num_object(cnts)
f.calc_perimeter(cnts)

#cnt = cnts[0]
#eclipse = cv.fitEllipse(cnt)
#cv.ellipse(erodila, ellipse, (0,255,0), 2)
'''
# define range of blue color in HSV
sensitivity = 5
lower_blue = np.array([60 - sensitivity, 50, 50])
upper_blue = np.array([60 + sensitivity, 255, 255])
# Threshold the HSV image to get only blue colors
mask = cv.inRange(image, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv.bitwise_and(image,image, mask= mask)

f.show_image("MASK", mask, False)
f.show_image("RES", res, False)
'''


'''
edged = cv.Canny(imgaus, 50, 100)
f.show_image("Edged", edged, False)
edged = cv.dilate(edged, None, iterations=1)
#edged = cv.erode(edged, None, iterations=1)
f.show_image("erode and dilate", edged, True)
'''