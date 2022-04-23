import src.mafunction as f
from src.segmentation import *
import src.segmentation as s

# Use command in terminal to use main.py

# Segmentation process
f.save_imgjson()

f.encode_imgjson()


#f.get_skeleton(thresh)
#f.fit_line()
f.fit_poly()
f.get_fish_length()

f.plot_curve2img()


f.show_imgjson()