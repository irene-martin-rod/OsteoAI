## PREPROCESSING IMAGES

import numpy as np
import cv2

#Preprocessing images
class PreprocessImage:
    """Class to resize, convert to grayscale, and normalize images with extended YOLO keypoints format."""

    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels

    def resize_images_and_labels(self, size=(640, 640)):
        '''Function to resize images and adjust labels with extended keypoints.'''
        resized_images = []
        resized_labels = {}

        for img, name in zip(self.images, self.labels.keys() if self.labels else [None] * len(self.images)):
            # Original image dimensions
            h, w = img.shape[:2]
            # Resize the image
            resized_img = cv2.resize(img, size)
            resized_images.append(resized_img)

            # Adjust labels if available
            if self.labels and name in self.labels:
                new_labels = []
                for label in self.labels[name]:
                    class_id = label[0]
                    # Ajustamos cada punto clave en pares (x, y) a las nuevas dimensiones
                    keypoints = []
                    for i in range(1, len(label), 2):  # Iniciamos en 1 para saltar el class_id
                        kp_x = label[i] * size[0] / w   # Ajuste del x
                        kp_y = label[i + 1] * size[1] / h  # Ajuste del y
                        keypoints.extend([kp_x, kp_y])

                    # Combina el class_id con los puntos clave ajustados
                    new_label = [class_id] + keypoints
                    new_labels.append(new_label)

                resized_labels[name] = new_labels

        return resized_images, resized_labels
    
    def convert_to_grayscale(self):
        '''Function to convert images to grayscale. Checks if the image is already grayscale, and converts if not.'''
        grayscale_images = []
        for img in self.images:
            if len(img.shape) == 2:
                grayscale_images.append(img)  # Already grayscale, no conversion needed
            else: 
                grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                grayscale_images.append(grayscale_img)
        return grayscale_images
    
    def normalize_images(self):
        '''Function to normalize pixel values to the range [0, 1].'''
        normalized_images = []
        for img in self.images:
            # Convert to array if necessary
            img_array = np.array(img) if not isinstance(img, np.ndarray) else img
            # Normalize to [0, 1] range
            normalized_img = img_array / 255.0
            normalized_images.append(normalized_img)
        return normalized_images