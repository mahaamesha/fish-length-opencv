import argparse

# To run in command line
# python main.py -i img/<filename>.jpg -w <value>
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='img/lele.jpg', help="path to the input image")
ap.add_argument("-w", "--width", default=1.0, type=float, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())