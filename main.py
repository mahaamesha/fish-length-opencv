from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import numpy as np
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
sensitivity = 20
lower_blue = np.array([60 - sensitivity, 50, 50])
upper_blue = np.array([60 + sensitivity, 255, 255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(image,image, mask= mask)

#gray = cv2.GaussianBlur(gray, (7, 7), 0)
#gray = cv2.GaussianBlur(gray, (1, 1), 0)

#edged = cv2.Canny(gray, 50, 100)
#show_image("Edged", edged, False)
#edged = cv2.dilate(edged, None, iterations=1)
#edged = cv2.erode(edged, None, iterations=1)
#show_image("erode and dilate", edged, True)

show_image("ORI", image, False)
show_image("GRAY", hsv, False)
show_image("MASK", mask, False)
show_image("RES", res, False)