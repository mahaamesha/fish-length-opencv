# I want to run this program on linux
import os

path = os.path.dirname(__file__)

# path that I want to delete
# path relative to src/ folder
add_dir = ['../imgcv/', '../bin/']

for dir in add_dir:
    path = os.path.join(path, dir)

    # Delete all files in dir, except final file
    if 'imgcv' in dir:
        os.system('rm -v !("final.jpg")')
    elif 'bin' in dir:
        os.system('rm -v !("final.bin")')