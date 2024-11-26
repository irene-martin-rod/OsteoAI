#Creating new directories to organized data as train, validation and test datasets. Inside this folder data are organized by fracture/no fracture

import os


import os

def create_directories(base_path):
    '''Function to create new directories to organize data as train, validation, and test.
    Inside these folders, data are organized by fracture/no fracture.
    Parameters:
    - base_path: the source directory where we want to create the new ones
    '''

    # Paths for the main directories (train, val, test)
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "val")
    test_dir = os.path.join(base_path, "test")

    # Check if main directories exist, and create them if not
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Main directories created:")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Test directory: {test_dir}")

    # Paths for the subdirectories (fracture, nofracture)
    train_fracture_dir = os.path.join(train_dir, "1_fracture")
    train_nofracture_dir = os.path.join(train_dir, "0_nofracture")
    val_fracture_dir = os.path.join(val_dir, "1_fracture")
    val_nofracture_dir = os.path.join(val_dir, "0_nofracture")
    test_fracture_dir = os.path.join(test_dir, "1_fracture")
    test_nofracture_dir = os.path.join(test_dir, "0_nofracture")

    # Print the paths of the subdirectories to check
    print(f"Creating subdirectories:")
    print(f"Train fracture directory: {train_fracture_dir}")
    print(f"Train nofracture directory: {train_nofracture_dir}")
    print(f"Val fracture directory: {val_fracture_dir}")
    print(f"Val nofracture directory: {val_nofracture_dir}")
    print(f"Test fracture directory: {test_fracture_dir}")
    print(f"Test nofracture directory: {test_nofracture_dir}")

    # Create subdirectories
    os.makedirs(train_fracture_dir, exist_ok=True)
    os.makedirs(train_nofracture_dir, exist_ok=True)
    os.makedirs(val_fracture_dir, exist_ok=True)
    os.makedirs(val_nofracture_dir, exist_ok=True)
    os.makedirs(test_fracture_dir, exist_ok=True)
    os.makedirs(test_nofracture_dir, exist_ok=True)

    print("Directories created successfully.")
    return None