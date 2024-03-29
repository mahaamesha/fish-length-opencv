import json
import cv2 as cv
import imutils
from matplotlib import markers, pyplot as plt
import numpy as np
import src.command as cmd
import os
from src.printlog import printlog

# SETUP PARAMETER
param = []
def config_segmentation(param=param, configChoice=0):    # import tmp/config.json
    path = os.path.join(os.path.dirname(__file__), str('../tmp/config.json'))
    with open(path, 'r') as fp:
        data = json.load(fp)
        for val in data.values():
            if val['_id'] == configChoice:
                param.append( (val['_blurSize'][0], val['_blurSize'][1]) )  # (x,y)
                param.append( (val['_threshVal'][0], val['_threshVal'][1]) )
                param.append( (val['_kernelSize'][0], val['_kernelSize'][1]) )
                param.append( (val['_erodilaIter'][0], val['_erodilaIter'][1]) )
                param.append( (val['_cannyThresh'][0], val['_cannyThresh'][1]) )
                
                printlog( str('Use segmentation config_' + str(val['_id'])).ljust(37, '.') + str('Done').rjust(5,' ') )
    #print(param)
    printlog()
    return param

config_segmentation()
#config_segmentation(param=s.param, configChoice=0)
    # 0 (51, 51), blurSize
    # 1 (0, 255), threshVal
    # 2 (3, 3),   kernelSize
    # 3 (3, 1),   erodilaIter
    # 4 (0, 255)  cannyThresh
# (END) SETUP PARAMETER


# IMAGE SEGMENTATION PROCESS
image_path = cmd.args["image"]
#image = cv.imread(image_path)            # Read the image
image = plt.imread(image_path)            # Read the image
cvt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)     # Convert object color to GRAY
imgaus = cv.GaussianBlur(cvt, (param[0][0],param[0][1]), 0)      # Blur using Gaussian Method to ignore noise

# Set BACKGROUND to Black and OBJECT to White
ret, thresh = cv.threshold(imgaus, param[1][0], param[1][1], cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Precise boundary using ERROSION & DILATION
kernel = np.ones( param[2], np.uint8 )
erodila = cv.erode(thresh, kernel, iterations=param[3][0])
erodila = cv.dilate(erodila, kernel, iterations=param[3][1])


# WATERSHED to improve contour detection
# Remove noise
opening = cv.morphologyEx(erodila, cv.MORPH_CLOSE, kernel, iterations=3)
# Sure background area
sure_bg = cv.dilate(erodila, kernel, iterations=11)
# Find sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
ret, sure_fg = cv.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
# Find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Mark label
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure_bg is not 0, but 1
markers = markers + 1
# Mark the region of unknown with 0
markers[unknown == 255] = 0
# Apply watershed
watershed_result = image.copy()
markers = cv.watershed(watershed_result, markers)
# Mark boundary region with -1
watershed_result[markers == -1] = [0, 255, 0]

smooth_border = cv.dilate(sure_fg, kernel, iterations=11)

# Detect the edge from BLACK WHITE img
edged = cv.Canny(smooth_border, param[4][0], param[4][1])

# Extract object only with black background
res = cv.bitwise_and(image, image, mask=smooth_border)

# (END) IMAGE SEGMENTATION PROCESS


# FIND CONTOUR
# Find many contour from BLACK WHITE img
cnts = cv.findContours(smooth_border, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

contoured = image.copy()
# Draw all contours in an img
contoured = cv.drawContours(contoured, cnts, -1, (0,255,0), 2)

# for ploting curve
final = contoured.copy()

# Check the '_id' in images.json to call it
# After finalization, only use final img
list_img = [
    #[_showflag, _encodeflag, _title, _var]
    [0, 0, 'image', image],
    [0, 0, 'cvt', cvt],
    [0, 0, 'imgaus', imgaus],
    [0, 0, 'thresh', thresh],
    [0, 0, 'erodila', erodila],
    [0, 0, 'sure_bg', sure_bg],
    [0, 0, 'dist_transform', dist_transform],
    [0, 0, 'sure_fg', sure_fg],
    [0, 0, 'unknown', unknown],
    [0, 0, 'smooth_border', smooth_border],
    [0, 0, 'watershed_result', watershed_result],
    [0, 0, 'edged', edged],
    [0, 0, 'res', res],
    [0, 0, 'contoured', contoured],
    [0, 1, 'final', final]
]