#Creating new directories to organized data as train, validation and test datasets. Inside this folder data are organized by fracture/no fracture

import os


def create_directories(base_path):

    '''Function to create new directories to organize data as train, validation and test. Inside these folders, data are organized by fracture/no fracture
    Parameters:
    - base_path: the source directory where we wante create the new ones
    '''

    # Creating main directories
    train_dir = os.path.join(base_path, "train")
    os.mkdir(train_dir)
    val_dir = os.path.join(base_path, "val")
    os.mkdir(val_dir)
    test_dir = os.path.join(base_path, "test")
    os.mkdir(test_dir)

    # Creating subdirectories for each category
    train_fracture_dir = os.path.join(train_dir, "fracture")
    os.mkdir(train_fracture_dir)
    train_nofracture_dir = os.path.join(train_dir, "nofracture")
    os.mkdir(train_nofracture_dir)
    val_fracture_dir = os.path.join(val_dir, "fracture")
    os.mkdir(val_fracture_dir)
    val_nofracture_dir = os.path.join(val_dir, "nofracture")
    os.mkdir(val_nofracture_dir)
    test_fracture_dir = os.path.join(test_dir, "fracture")
    os.mkdir(test_fracture_dir)
    test_nofracture_dir = os.path.join(test_dir, "nofracture")
    os.mkdir(test_nofracture_dir)

    return None
