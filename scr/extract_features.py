###EXTRACT FEATURES TO IMAGES

import numpy as np

def extract_features(generator, model):
    """
    Extracting features and corresponding labels from a dataset using a pre-trained model.

    Parameters:
        generator (DirectoryIterator): Data generator that yields batches of images and labels.
        model (keras.Model): Pre-trained model used for feature extraction.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Extracted feature embeddings from the images, concatenated for all batches.
            - labels (numpy.ndarray): Corresponding labels for the images, concatenated for all batches.
    """
    features = []
    labels = []

    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)  # Extract embeddings
        features.append(features_batch)
        labels.append(labels_batch)
        if len(features) * generator.batch_size >= generator.samples:
            break  # Avoid infinite loops in data generators
    return np.concatenate(features), np.concatenate(labels)