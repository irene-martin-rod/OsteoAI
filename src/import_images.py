###IMPORT IMAGES

import tensorflow as tf
import os

def create_image_dataset(base_path, subset="training", validation_split=0.1, color_mode="rgb", image_size=(256, 256), batch_size=20, labels = "inferred", 
                         label_mode="binary", seed=42):
    '''
    Creates an image dataset from a directory using image_dataset_from_directory.

    Parameters:
        base_path (str): Path to the root directory of the dataset.
        subset (str): Subset of data to load. Can be 'training' or 'validation'.
        validation_split (float): Percentage of data reserved for validation. Default is 0.1.
        color_mode (str): 'rgb' for color images or 'grayscale' for grayscale. Default is 'rgb'.
        image_size (tuple): Size to resize images to. Default is 256*256
        batch_size (int): Number of images per batch. Default is 20.
        labels (str): By default is 'inferred': labels are generated from the directory structure
        label_mode (str): Labeling mode ('categorical', 'binary', 'int', or None). Default is 'binary'.
        seed (int): Seed for reproducibility. Default is 42.

    Returns:
        tf.data.Dataset: A dataset ready to use with the tf.data API.
    '''
    return tf.keras.preprocessing.image_dataset_from_directory(
        base_path,
        color_mode=color_mode,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        labels = labels,
        label_mode=label_mode
    )