import src.mafunction as f

# Use command in terminal to use main.py
#python main.py -i img/lele.jpg -w 1

# Segmentation process
f.save_imgjson()

f.encode_imgjson()


#f.get_skeleton(thresh)
#f.fit_line()
f.fit_poly()
f.get_fish_length()

f.plot_curve2img()


f.show_imgjson()
f.decode_imgjson()