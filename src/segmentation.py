import cv2 as cv
import imutils
import numpy as np
import src.command as cmd
from os.path import exists as file_exists

# IMAGE SEGMENTATION PROCESS
image_path = cmd.args["image"]
image = cv.imread(image_path)            # Read the image
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

# for ploting curve
final = contoured.copy()

# Check the '_id' in images.json to call it
list_img = [
    image,
    cvt,
    imgaus,
    thresh,
    erodila,
    edged,
    res,
    contoured,
    final
]


# Export to json file --> in main.py


# Call this function in main.py
#bulk_showORsave_img(func=show_img()))
#bulk_showORsave_img(func=save_img()))