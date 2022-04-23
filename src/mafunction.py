from ast import arguments
from tkinter import Y
import cv2 as cv
from matplotlib import contour
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os

import src.segmentation as s


def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

def clear_json_file(title):
    filename = title + '.json'
    path = 'tmp/' + filename
    with open(path,'w') as f:
        f.close()

def write_json(title, dict):
    filename = title + '.json'
    path = 'tmp/' + filename

    with open(path,'r+') as file:
        file_data = json.load(file)
        key = file_data.keys()
        file_data[key].append(dict)
        file.seek(0)
        json.dump(file_data, file, indent=4)


def points2json(title, dict):
    # dict = {
    #   'x': [...], 
    #   'y': [...]
    # }
    # final = {
    #   'curve_0':{
    #       'x': [...],
    #       'y': [...]
    #   }, ...
    filename = title + '.json'
    path = 'tmp/' + filename

    with open(path, 'r') as fr:
        if (not is_file_empty(path)):
            final_dict = json.load(fr)      # points.json should be initialized first & not empty
        else: final_dict = {}
    
        # if more than num_of_contours, delete old data
        if (len(final_dict) > len(s.cnts)):
            for i in range(len(final_dict)-1):
                del final_dict[str('curve_'+str(i))]
    final_dict[str('curve_'+str(len(final_dict)))] = dict

    
    with open(path, 'w') as fp:
        json.dump(final_dict, fp, indent=4)
    #print('Export', filename, '... Done')



def show_image(title, image, destroy_all=False):
    cv.imshow(title, image)
    cv.waitKey(0)
    if destroy_all:
        cv.destroyAllWindows()


def json_show_image(jsonfile='images.json'):
    path = 'tmp/' + jsonfile
    f = open(path)
    data = json.load(f)
    # see the structure in file: is_show_image.json
    for val in data.values():
        for i in range(len(val)):
            #print('>>>', val[i]['_title'])
            if (val[i]['_flag'] == 1):
                print('tes')
                #show_image(title=val[i]['_title'], image=val[i]['_variable'])
    f.close


def save_img(title, img):
    filename = title + '.jpg'
    path = 'imgcv/' + filename
    status = cv.imwrite(path, img)
    # folder should be initialized first
    print('Image written to', path, '...', status)


def save_img2json(jsonfile='images.json'):
    path = 'tmp/' + jsonfile
    f = open(path)
    data = json.load(f)
    # see the structure in file: is_show_image.json
    for val in data.values():
        for i in range(len(val)):
            print('>>>', val[i]['_variable'])
            #save_img(val[i]['_variable'])
    print('Saving *.jpg to imgcv... Done')


def calc_perimeter(cnts=s.cnts):
    perimeter = []
    for i in cnts:
        arc_length = cv.arcLength(i, True)
        perimeter.append(arc_length)
    print("perimeter:", "{:.2f}".format(perimeter))

    return perimeter


def calc_area(cnts=s.cnts):
    area = []
    for cnt in cnts:
        area.append( cv.contourArea(cnt) )
    print("area:", area)

    return area


def num_moments(cnts=s.cnts):
    moments = []
    for cnt in cnts:
        moments.append( cv.moments(cnt) )
    print("moment:", moments)

    return moments


def get_skeleton(img=s.erodila):
    # Draw skeleton of banana on the mask
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret, img = cv.threshold(img,5,255,0)
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

    cv.drawContours(img, skel, -1, (0, 0, 255), 2)
    show_image("skeleton", skel, False)


def fit_line(cnts=s.cnts, image=s.image):
    for cnt in cnts:
        rows, cols = image.shape[:2]
        [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv.line(image, (cols-1,righty),(0,lefty),(0,255,0),2)
    show_image("fit", image, False)


# To draw coordinate for all edge points
approx_factor = 5 * 1/1000  # More smaller, more points detected
def edge_points(cnts=s.cnts, image=s.image):
    font = cv.FONT_HERSHEY_COMPLEX

    # Going through every contours found in the image.
    for cnt in cnts :
        approx = cv.approxPolyDP(cnt, approx_factor * cv.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv.drawContours(image, [approx], 0, (0, 0, 255), 5) 
    
        # Flaten the array
        n = approx.ravel()
        #print("Flaten array:", n)   # For future: store in json file
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
    
                # String containing the co-ordinates.
                string = str(x) + " " + str(y) 
    
                if(i == 0):
                    # text on topmost co-ordinate.
                    cv.putText(image, "Arrow tip", (x, y), font, 0.5, (255, 0, 0)) 
                else:
                    # text on remaining co-ordinates.
                    cv.putText(image, string, (x, y), font, 0.5, (0, 255, 0)) 
            i = i + 1


# Length of fitting curve result
def curve_length(dict_points):  # distance of 2 points: A and B
    sum = 0
    length = len( dict_points['x'] )
    for i in range(length-1):
        A = ( dict_points['x'][i], dict_points['y'][i] )
        B = ( dict_points['x'][i+1], dict_points['y'][i+1] )
        dx = abs(B[0] - A[0])   # |x2-x1|
        dy = abs(B[1] - A[1])   # |y2-y1|

        dl = math.sqrt(dy*dy + dx*dx)

        sum += dl
    
    #print('Fish Length:', "{:.2f}".format(sum))
    return sum

    # Get curve length for every contour, then add to dictionary
    #fish_length[str(len(fish_length))] = curve_length(points)


# measure fish length with data from points.json
def get_fish_length():
    fish_length = {}
    with open('tmp/points.json', 'r') as fp:
        data = json.load(fp)
        for key in data.keys(): # key are "curve_0", "curve_1", ...
            A = fish_length[str('fish_'+ key[8:])] = {}
            A.update( {'length': curve_length(data[key])} )
    
    # Write it to fish_length.json
    with open('tmp/fish_length.json', 'w') as fp:
        json.dump(fish_length, fp, indent=4)

    print('Measure fish length... Done')




points = {}
def fit_poly(cnts=s.cnts, showPlot=False):
    # Get all approx points in a contour
    # approx_factor defined in 'edge_points()'
    
    clear_json_file('points')

    for cnt in cnts :
        approx = cv.approxPolyDP(cnt, approx_factor * cv.arcLength(cnt, True), True)   # 0.005 : more lower more points detected
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
        myline = np.linspace(min(x), max(x), 20)    # step 20 default

        # get all curve fitting coordinate
        list_x = myline.tolist()
        list_y = []
        for i in list_x:
            list_y.append( mymodel(i) )
        points["x"] = list_x
        points["y"] = list_y

        # Export points to json
        points2json('points', points)

        # Draw curve to image
        #for 

        if showPlot:
            plt.scatter(x, y)
            plt.plot(myline, mymodel(list_x))
            plt.show()
    print('Curve fitting for every contour... Done')



def plot_curve2img(title='lele'):
    filename = title + '.jpg'
    path = 'img/' + filename

    plt.rcParams["figure.autolayout"] = True

    im = plt.imread(path)
    fig, ax = plt.subplots()
    im = ax.imshow(im)

    with open('tmp/points.json', 'r') as fp:
        data = json.load(fp)
        
        for key in data.keys():     # key: curve_0, curve_1, ...
            # data['x] have been sorted automaticaly
            x = data[key]['x']
            y = data[key]['y']
            
            ax.plot(x, y, ls='dotted', linewidth=2, color='red')
        print()
    
    plt.savefig('imgcv/final.jpg')
    plt.show()
    print('Plot Curve to img.... Done')