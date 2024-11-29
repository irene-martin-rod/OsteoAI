###PREPROCESSING IMAGES AND MAKING DATA AUGMENTATION

import tensorflow as tf



def preprocess_image(image, label, rescale=1./255, rotation_range=0, width_shift_range=0, height_shift_range=0, zoom_range=0, horizontal_flip=False):
    '''
    Applies preprocessing transformations to the image.

    Arguments:
        image: Input image tensor.
        label: Corresponding label tensor.
        rescale: Normalizes image values to [0, 1]. Default is 1./255.
        rotation_range: Max degree for random rotation. Default is 0.
        width_shift_range: Fraction of total width for horizontal translation. Default is 0.
        height_shift_range: Fraction of total height for vertical translation. Default is 0.
        zoom_range: Fraction range for random zoom. Default is 0.
        horizontal_flip: Whether to randomly flip images horizontally. Default is False.

    Returns:
        (image, label): Preprocessed image and label tensors.
    '''
    
    # Normalize the image: Converts the pixel values to [0, 1] if they are not already.
    # This is important because neural networks typically work better with values between [0, 1].
    # `tf.image.convert_image_dtype` automatically scales pixel values to the range [0, 1] when converting to float32.
    image = tf.image.convert_image_dtype(image, tf.float32)  
    image = image * rescale  # Apply additional scaling (default rescale = 1./255)

    # Random rotation: If rotation_range > 0, apply random rotations to the image.
    if rotation_range:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)  # Random rotation index (0, 1, 2, or 3)
        image = tf.image.rot90(image, k=k)  # Apply the rotation: 0 -> no rotation, 1 -> 90 degrees, 2 -> 180 degrees, etc.

    # Random horizontal shift: If width_shift_range > 0, apply random horizontal translation to the image.
    if width_shift_range:
        width_shift = tf.cast(tf.shape(image)[1] * width_shift_range, tf.int32)  # Convert to integer
        image = tf.image.resize_with_crop_or_pad(image,
                                                 target_height=tf.shape(image)[0], 
                                                 target_width=tf.shape(image)[1] - width_shift)  # Apply width shift by resizing

    # Random vertical shift: If height_shift_range > 0, apply random vertical translation to the image.
    if height_shift_range:
        height_shift = tf.cast(tf.shape(image)[0] * height_shift_range, tf.int32)  # Convert to integer
        image = tf.image.resize_with_crop_or_pad(image, 
                                                 target_height=tf.shape(image)[0] - height_shift,
                                                 target_width=tf.shape(image)[1])  # Apply height shift by resizing

    # Random zoom: If zoom_range > 0, apply random zoom to the image.
    if zoom_range:
        scales = [1 - zoom_range, 1 + zoom_range]  # Defines the range for scaling: [1 - zoom_range, 1 + zoom_range]
        scale = tf.random.uniform([], scales[0], scales[1])  # Random scale factor within the range
        image = tf.image.resize(image, tf.cast(tf.shape(image)[0:2] * scale, tf.int32))  # Apply the zoom by resizing the image

    # Random horizontal flip: If horizontal_flip is True, apply random flipping of the image along the vertical axis.
    if horizontal_flip:
        image = tf.image.random_flip_left_right(image)  # Randomly flip the image horizontally

    # Return the preprocessed image and its label
    return image, label




def apply_preprocessing(dataset, rescale=1./255, rotation_range=0, width_shift_range=0, height_shift_range=0, 
                        zoom_range=0, horizontal_flip=False):
    '''
    Applies preprocessing transformations to a dataset.

    Arguments:
        dataset: The tf.data.Dataset containing images and their corresponding labels.
        rescale: Normalizes the pixel values of images to [0, 1]. Default is 1./255.
        rotation_range: The range (in degrees) within which to apply random rotations. Default is 0 (no rotation).
        width_shift_range: Fraction of total width for random horizontal translation. Default is 0 (no shift).
        height_shift_range: Fraction of total height for random vertical translation. Default is 0 (no shift).
        zoom_range: Fraction range for random zoom. Default is 0 (no zoom).
        horizontal_flip: Boolean flag to indicate if horizontal flipping should be applied. Default is False.

    Returns:
        tf.data.Dataset: The processed dataset with applied augmentations and preprocessing.
    '''

    # Apply transformations to each image in the dataset
    return dataset.map(
        lambda image, label: preprocess_image(image, label, rescale, rotation_range, width_shift_range, height_shift_range, zoom_range, horizontal_flip),
        num_parallel_calls=tf.data.AUTOTUNE  # Use automatic parallel calls to optimize performance
    )

