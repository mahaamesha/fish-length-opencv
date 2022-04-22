from tkinter import Y
import cv2 as cv
from cv2 import LINE_AA
from matplotlib import contour
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

perimeter = []
def calc_perimeter(cnts):
    for i in cnts:
        arc_length = cv.arcLength(i, True)
        perimeter.append(arc_length)
    print("perimeter:", perimeter)

area = []
def num_object(cnts):
    for cnt in cnts:
        area.append( cv.contourArea(cnt) )
    print("area:", area)

moments = []
def num_moments(cnts):
    for cnt in cnts:
        moments.append( cv.moments(cnt) )
    print("moment:", moments)


def get_skeleton(erodila):
    img = erodila
    # Draw skeleton of banana on the mask
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret,img = cv.threshold(img,5,255,0)
    element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv.erode(img,element)
        temp = cv.dilate(eroded,element)
        temp = cv.subtract(img,temp)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy() 
        zeros = size - cv.countNonZero(img)
        if zeros==size: done = True
    kernel = np.ones((2,2), np.uint8)
    skel = cv.dilate(skel, kernel, iterations=1)
    skeleton_contours, _ = cv.findContours(skel, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #import imutils
    #skeleton_contours = imutils.grab_contours(skeleton_contours)
    #largest_skeleton_contour = max(skeleton_contours, key=cv.contourArea)

    cv.drawContours(img, skel, -1, (0, 0, 255), 2, LINE_AA)
    show_image("skeleton", skel, False)

def fit_line(cnts, image):
    for cnt in cnts:
        rows, cols = image.shape[:2]
        [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv.line(image, (cols-1,righty),(0,lefty),(0,255,0),2)
    show_image("fit", image, False)



def edge_points(cnts, image):
    font = cv.FONT_HERSHEY_COMPLEX

    # Going through every contours found in the image.
    for cnt in cnts :
    
        approx = cv.approxPolyDP(cnt, 0.005 * cv.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv.drawContours(image, [approx], 0, (0, 0, 255), 5) 
    
        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel()
        print("NNNNNNNNNNNNNNNNNNNNNNNN:", n)
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
    
                # String containing the co-ordinates.
                string = str(x) + " " + str(y) 
    
                if(i == 0):
                    # text on topmost co-ordinate.
                    cv.putText(image, "Arrow tip", (x, y),
                                    font, 0.5, (255, 0, 0)) 
                else:
                    # text on remaining co-ordinates.
                    cv.putText(image, string, (x, y), 
                            font, 0.5, (0, 255, 0)) 
            i = i + 1


import math
def curve_length(dict_points):  # distance of 2 points
    sum = 0
    length = len( dict_points['x_curve'] )
    for i in range(length-1):     # A, B is array with same length
        A = ( dict_points['x_curve'][i], dict_points['y_curve'][i] )
        B = ( dict_points['x_curve'][i+1], dict_points['y_curve'][i+1] )
        dx = abs(B[0] - A[0])
        dy = abs(B[1] - A[1])

        dl = math.sqrt(dy*dy + dx*dx)

        sum += dl
    
    print('Fish Length:', sum)
    return sum


points = {}
fish_length = {}
def fit_poly(cnts):
    # Get all approx points in a contour
    for cnt in cnts :
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)   # 0.005 : more lower more points detected
        n = approx.ravel()

        # Extract x, y coordinate
        x = []
        y = []
        for i in range(len(n)):
            # Get x and y
            if (i % 2 == 0):
                x.append(n[i])
            else:
                y.append(n[i])
        
        mymodel = np.poly1d(np.polyfit(x, y, 3))    # 3th degree polynomial
        myline = np.linspace(min(x), max(x), 30)

        # get all curve fitting coordinate
        list_x_curve = myline.tolist()
        list_y_curve = []
        for i in list_x_curve:
            list_y_curve.append( mymodel(i) )
        points["x_curve"] = list_x_curve
        points["y_curve"] = list_y_curve
        print("POOOINNNTSSS", points)

        # get curve length for every contour
        fish_length[str(len(fish_length))] = curve_length(points)

        
        plt.scatter(x, y)
        plt.plot(myline, mymodel( myline.tolist() ))
        plt.show()
