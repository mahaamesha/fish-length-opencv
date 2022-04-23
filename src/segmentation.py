import cv2 as cv
import imutils
import numpy as np
import src.command as cmd

# IMAGE SEGMENTATION PROCESS
image = cv.imread(cmd.args["image"])            # Read the image
cvt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)     # Convert object color to GRAY
imgaus = cv.GaussianBlur(cvt, (51, 51), 0)      # Blur using Gaussian Method to ignore noise

# Set BACKGROUND to Black and OBJECT to White
ret, thresh = cv.threshold(imgaus,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Precise boundary using ERROSION & DILATION
kernel = np.ones( (3,3), np.uint8 )
erodila = cv.erode(thresh, kernel, iterations=3)
erodila = cv.dilate(erodila, kernel, iterations=1)

# Detect the edge from BLACK WHITE img
edged = cv.Canny(erodila, 0, 255)

# Extract object only with black background
res = cv.bitwise_and(image,image, mask=erodila)

# (END) IMAGE SEGMENTATION PROCESS


# IMAGE PROCESSING
# Find many contour from BLACK WHITE img
cnts = cv.findContours(erodila, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("Total number of contours: ", len(cnts))

contoured = image.copy()
# Draw all contours in an img
cv.drawContours(contoured, cnts, -1, (0,255,0), 2)
#f.show_image("image_copy", image, False)


'''
f.show_image("ORIGINAL", image) # original
f.show_image("GRAY COLOR", cvt) # gray object
f.show_image("GAUSSIAN", imgaus)    # blur img
f.show_image("THRESHOLD", thresh)   # set black & white
f.show_image("EROSION & DILATION", erodila) # precise boundary from black & white img
f.show_image("EDGE", edged) # edge only from erodila
f.show_image("RESULT", res) # extract result: catfish only in black background
'''