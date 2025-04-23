#Copy images from one directory to another one

import os
import shutil

def copy_images(origen_directory, new_directory, start, end):

    '''Function to copy files from a source directory (origen_directory) to a destination directory (new_directory) indised a specific range.
     Parameters:
     - origen_dir: Source directory
     - new_directory: Destination directory where all files are copied
     - start: the starting index of the range of files to copy
     - end: the ending index of the range of files to copy
     '''

    #List of all files in the origen directory inside a range
    fnames = [i for i in os.listdir(origen_directory)[start : end]]

    for fname in fnames:
        src = os.path.join(origen_directory ,fname)
        dst = os.path.join(new_directory, fname)
        shutil.copyfile(src,dst)

    print(f"{len(fnames)} copied from {origen_directory} to {new_directory}")