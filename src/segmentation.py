import json
import cv2 as cv
import imutils
import numpy as np
import src.command as cmd


# SETUP PARAMETER
param = []

#config_segmentation(param=s.param, configChoice=0)
    # 0 (51, 51), blurSize
    # 1 (0, 255), threshVal
    # 2 (3, 3),   kernelSize
    # 3 (3, 1),   erodilaIter
    # 4 (0, 255)  cannyThresh

def config_segmentation(param=param, configChoice=0):    # import tmp/config.json
    with open('tmp/config.json', 'r') as fp:
        data = json.load(fp)
        for val in data.values():
            if val['_id'] == configChoice:
                param.append( (val['_blurSize'][0], val['_blurSize'][1]) )  # (x,y)
                param.append( (val['_threshVal'][0], val['_threshVal'][1]) )
                param.append( (val['_kernelSize'][0], val['_kernelSize'][1]) )
                param.append( (val['_erodilaIter'][0], val['_erodilaIter'][1]) )
                param.append( (val['_cannyThresh'][0], val['_cannyThresh'][1]) )
                
                print( str('Use segmentation config_' + str(val['_id'])).ljust(37, '.') + str('Done').rjust(5,' ') )
    #print(param)
    print()
    return param

config_segmentation()
# (END) SETUP PARAMETER


# IMAGE SEGMENTATION PROCESS
image_path = cmd.args["image"]
image = cv.imread(image_path)            # Read the image
cvt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)     # Convert object color to GRAY
imgaus = cv.GaussianBlur(cvt, (param[0][0],param[0][1]), 0)      # Blur using Gaussian Method to ignore noise

# Set BACKGROUND to Black and OBJECT to White
ret, thresh = cv.threshold(imgaus, param[1][0], param[1][1], cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Precise boundary using ERROSION & DILATION
kernel = np.ones( param[2], np.uint8 )
erodila = cv.erode(thresh, kernel, iterations=param[3][0])
erodila = cv.dilate(erodila, kernel, iterations=param[3][1])

# Detect the edge from BLACK WHITE img
edged = cv.Canny(erodila, param[4][0], param[4][1])

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