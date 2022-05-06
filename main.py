import src.mafunction as f
import src.segmentation as s

# Use command in terminal to use main.py
#python main.py -i img/lele.jpg -w 1

# Image segmentation process
# Have been imported in mafunction.py
f.generate_imagesjson()
f.save_imgjson()

f.encode_imgjson()


#f.get_skeleton(thresh)
#f.fit_line()
f.fit_poly()
f.get_fish_length()
#f.validate_fish_length()

f.plot_curve2img()


f.show_imgjson()
print("Total number of contours: ", len(s.cnts))
#f.decode_imgjson()