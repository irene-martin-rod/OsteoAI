###EXTRACT FEATURES TO IMAGES

import numpy as np


def extract_features(model, dataset):
    """
    Extract features and corresponding labels from a dataset using a pre-trained model.

    Parameters:
        model (keras.Model): Pre-trained model used for feature extraction.
        dataset (tf.data.Dataset): Dataset providing input images and labels.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Extracted feature embeddings from the images.
            - labels (numpy.ndarray): Corresponding labels for the images.
    """
    features = []
    labels = []

    for batch in dataset.as_numpy_iterator():
        inputs_batch, labels_batch = batch
        # Extract features for the batch
        features_batch = model.predict(inputs_batch, verbose=0)
        features.append(features_batch)
        labels.append(labels_batch)

    # Concatenate all features and labels into single arrays
    return np.concatenate(features), np.concatenate(labels)