import src.mafunction as f

# Use command in terminal to use main.py
#python main.py -i img/lele.jpg -w 1

# Image segmentation process
# Have been imported in mafunction.py

# Initialize images.json
f.generate_imagesjson()

# Save the img to imgcv/
f.save_imgjson()

# Create curve for each contour and save their points to points.json
f.fit_poly()

# Calculate fish length from each curve
f.get_fish_length()

# Validating
num_fish = f.validate_num_fish()
fishlength = f.validate_fishlength()

# Save datetime, num_fish, avg_fishlength to result.json
f.generate_resultjson()
# Plot curve to img with the result as title. Then, save final.jpg
f.plot_curve2img()
f.numbering_curve()
# Swow img whose _flagshow == 1
f.show_imgjson()
# Encode img whose _encodeflag == 1
f.encode_imgjson()

f.printlog("Total number of fish:", num_fish)
f.printlog("Average of fish length:", fishlength, end='\n\n')

#f.decode_imgjson()