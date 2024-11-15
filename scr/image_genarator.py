###IMAGE GENERATORR

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_datagen(rescale = 1./255, rotation_range = 0, width_shift_range = 0, height_shift_range = 0, 
                   zoom_range = 0, horizontal_flip = False):
    """
    Creating an instance of ImageDataGenerator with the next parameters:
    
    Arguments:
        rescale: Normalized image with values 0-1.
        rotation_range: Rotation range in degrees. By default: 0
        width_shift_range: Horizontal traslation range in degrees. By default: 0
        height_shift_range: Verticaltraslation range in degrees. By default: 0
        zoom_range: Zoom range. By default: 0
        horizontal_flip: Activae/deactivate horizontal flip. By default: False
    
    Returns:
        An instance of ImageDataGenerator.
    """
    return ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip
    )


def create_generator(datagen, directory, target_size = (256, 256), batch_size = 20, class_mode = "binary", shuffle = True):
    """
    Creating a generator using flow_from_directory to load and preprocess images.
    
    Argumentss:
        datagen: Data generator instance
        directory: path to images.
        target_size: tuple with the size to resize images. By dafault: (256, 256)
        batch_size: Number of image by batch. By default 20
        class_mode: Classification type. By default: 'binary' Tipo de clasificación ('binary', 'categorical', etc.).
        shuffle: If the images are chosen randomly. By default: True
    
    Returns:
        An objet to data genarate.
    """
    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )