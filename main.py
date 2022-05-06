import src.mafunction as f
import src.segmentation as s

# Use command in terminal to use main.py
#python main.py -i img/lele.jpg -w 1

# Image segmentation process
# Have been imported in mafunction.py

# Initialize images.json
f.generate_imagesjson()

# Save & encode all img
f.save_imgjson()
f.encode_imgjson()

# Create curve for each contour and save their points to points.json
f.fit_poly()

# Calculate fish length from each curve
f.get_fish_length()

#f.validate_fish_length()
num_fish = f.validate_num_fish()

# Save datetime, num_fish, avg_fishlength to result.json
f.generate_resultjson()
# Plot curve to img with the result as title
f.plot_curve2img()
f.numbering_curve()
# Swow img whose _flag == 1
f.show_imgjson()

print("Total number of contours: ", num_fish)

#f.decode_imgjson()