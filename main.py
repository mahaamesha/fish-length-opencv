import src.mafunction as f
import src.segmentation as s

# Use command in terminal to use main.py
#python main.py -i img/lele.jpg -w 1

# Image segmentation process
# Have been imported in mafunction.py
f.save_imgjson()

f.encode_imgjson()


#f.get_skeleton(thresh)
#f.fit_line()
f.fit_poly()
f.get_fish_length()
#f.validate_fish_length()

f.plot_curve2img()


f.show_imgjson()
f.show_img('opening', s.opening)
f.show_img('sure_bg', s.sure_bg)
f.show_img('dist_traf', s.dist_transform)
f.show_img('sure_fg', s.sure_fg)
f.show_img('unknown', s.unknown)
f.show_img('water', s.watershed_result)
print("Total number of contours: ", len(s.cnts))
#f.decode_imgjson()