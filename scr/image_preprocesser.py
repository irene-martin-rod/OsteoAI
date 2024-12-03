###PREPROCESSING IMAGES AND MAKING DATA AUGMENTATION

import tensorflow as tf



def preprocess_image(image, label, rescale=1./255, rotation_range=0, width_shift_range=0, height_shift_range=0, zoom_range=0, 
                     horizontal_flip=False, output_size=None):
    '''
    Applies preprocessing transformations to the image with flexible resizing.

    Arguments:
        image: Input image tensor.
        label: Corresponding label tensor.
        rescale: Normalizes image values to [0, 1]. Default is 1./255.
        rotation_range, width_shift_range, height_shift_range, zoom_range: Preprocessing augmentations.
        horizontal_flip: Whether to randomly flip images horizontally. Default is False.
        output_size: (height, width) tuple. If None, keeps original size.

    Returns:
        (image, label): Preprocessed image and label tensors.
    '''
    # Normalize
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image * rescale, 0.0, 1.0)

    # Random rotation
    if rotation_range:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)

    # Random horizontal shift
    if width_shift_range:
        shift_width = tf.random.uniform([], -width_shift_range, width_shift_range) * tf.cast(tf.shape(image)[1], tf.float32)
        shift_width = tf.cast(shift_width, tf.int32)
        if shift_width < 0:
            image = tf.image.crop_to_bounding_box(image, 0, -shift_width, tf.shape(image)[0], tf.shape(image)[1] + shift_width)
        else:
            image = tf.image.crop_to_bounding_box(image, 0, 0, tf.shape(image)[0], tf.shape(image)[1] - shift_width)
        image = tf.image.resize_with_crop_or_pad(image, tf.shape(image)[0], tf.shape(image)[1])

    # Random vertical shift
    if height_shift_range:
        shift_height = tf.random.uniform([], -height_shift_range, height_shift_range) * tf.cast(tf.shape(image)[0], tf.float32)
        shift_height = tf.cast(shift_height, tf.int32)
        if shift_height < 0:
            image = tf.image.crop_to_bounding_box(image, -shift_height, 0, tf.shape(image)[0] + shift_height, tf.shape(image)[1])
        else:
            image = tf.image.crop_to_bounding_box(image, 0, 0, tf.shape(image)[0] - shift_height, tf.shape(image)[1])
        image = tf.image.resize_with_crop_or_pad(image, tf.shape(image)[0], tf.shape(image)[1])

    # Random zoom
    if zoom_range:
        scales = [1 - zoom_range, 1 + zoom_range]
        scale = tf.random.uniform([], scales[0], scales[1])
        new_size = tf.cast(scale * tf.cast(tf.shape(image)[0:2], tf.float32), tf.int32)
        image = tf.image.resize(image, new_size)
        if output_size:
            image = tf.image.resize_with_crop_or_pad(image, output_size[0], output_size[1])

    # Resize to output size if specified
    if output_size:
        image = tf.image.resize(image, output_size)

    # Random horizontal flip
    if horizontal_flip:
        image = tf.image.random_flip_left_right(image)

    return image, label


def apply_preprocessing(dataset, rescale=1./255, rotation_range=0, width_shift_range=0, height_shift_range=0, 
                        zoom_range=0, horizontal_flip=False, output_size=None):
    '''
    Applies preprocessing transformations to a dataset with flexible resizing.

    Arguments:
        dataset: The tf.data.Dataset containing images and labels.
        rescale, rotation_range, width_shift_range, height_shift_range, zoom_range, horizontal_flip: Preprocessing parameters.
        output_size: (height, width) tuple. If None, keeps original size.

    Returns:
        tf.data.Dataset: Preprocessed dataset.
    '''
    def preprocess_and_resize(image, label):
        image, label = preprocess_image(image, label, rescale, rotation_range, width_shift_range, height_shift_range, zoom_range, horizontal_flip, output_size)
        return image, label

    return dataset.map(preprocess_and_resize, num_parallel_calls=tf.data.AUTOTUNE)


