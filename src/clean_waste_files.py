# I want to delete all files in bin/ & imgcv/, 
# except final.bin & final.jpg

import os

def main():
    # path that I want to delete
    # path relative to src/ folder
    add_dir = ['../imgcv/', '../bin/', '../imgtest/']

    for dir in add_dir:
        working_path = os.path.dirname(__file__)
        path = os.path.join(working_path, dir)

        # Scan all file in dir
        list = []
        for (root, dirs, file) in os.walk(path):
            for f in file:
                if 'final' not in f:    # Save all to list, except final.format
                    list.append(f)

        # Delete all files in dir, except final file
        for fn in list:
            fnpath = str(path + fn)
            os.remove(fnpath)


if __name__ == '__main__':
    main()