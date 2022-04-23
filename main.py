from email.policy import default
from matplotlib import markers
from matplotlib.pyplot import get
from pyppeteer import defaultArgs
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2 as cv

import src.mafunction as f
from src.segmentation import *
import src.segmentation as s

# Use command in terminal to use main.py

# Segmentation process
idd = f.get_id_imgjson('image')
tit = 'image'
im = s.list_img[idd]
f.show_img(tit, im)

f.save_imgjson()

#f.edge_points()
#f.show_image("image_copy", image, False)
#f.json_show_image()
#f.bulk_showORsave_img() # default: save_img()
#f.save_img('final', s.contoured)
#f.show_image('final', s.contoured)

#f.encode_img('final')
#f.decode_img('final')
f.imgstr2json()


#f.get_skeleton(thresh)
#f.fit_line()
f.fit_poly()
f.get_fish_length()
f.plot_curve2img()

#f.calc_area(cnts)
#f.calc_perimeter(cnts)
#f.save_img2json()
#f.save_img('edged', edged)