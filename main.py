from email.policy import default
from cv2 import FILLED, LINE_AA, ellipse
from matplotlib import markers
from pyppeteer import defaultArgs
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2 as cv

import src.command as cmd

import src.mafunction as f

# Use command in terminal to use main.py

# Segmentation process
from src.segmentation import *


f.edge_points()
#f.show_image("image_copy", image, False)

#f.get_skeleton(thresh)
f.fit_line()
f.fit_poly()

#f.calc_area(cnts)
#f.calc_perimeter(cnts)
f.save_img2json()
#f.save_img('edged', edged)