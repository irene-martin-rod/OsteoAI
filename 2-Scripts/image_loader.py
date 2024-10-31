## LOADING IMAGES AND PLOTTING

import os
import cv2
import random
import matplotlib.pyplot as plt

#Loading and plotting images
class LoadImage:
    '''Class to load and visualize images''' 

    def __init__(self, image_path, label_path=None):
        self.image_path = image_path
        self.label_path = label_path  # Puede ser None si no hay etiquetas
        self.images = []
        self.names = []
        self.load_images()  # Cargar imágenes al iniciar la clase

    def load_images(self):
        '''Function to load images using OpenCV'''
        for root, dirs, files in os.walk(self.image_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        self.images.append(image)
                        self.names.append(file)

    def load_labels(self):
        '''Function to load labels'''
        if self.label_path is None:
            return {}

        labels = {}
        for file in os.listdir(self.label_path): 
            final_path = os.path.join(self.label_path, file)
            with open(final_path, 'r') as f: 
                label_data = [list(map(float, line.strip().split())) for line in f.readlines()]
                labels[file] = label_data
        return labels
    
    def plot_images_with_bboxes(self, num_images=2):
        '''Method to plot a random selection of images with bounding boxes if available'''
        labels = self.load_labels()

        if not self.images:
            print("No images found to display.")
            return

        num_images = min(num_images, len(self.images))
        selected_indices = random.sample(range(len(self.images)), num_images)

        for idx in selected_indices:
            img = self.images[idx]
            img_name = self.names[idx]
            img_labels = labels.get(img_name.replace('.jpg', '.txt'), None)

            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if img_labels:
                height, width, _ = img.shape

                for label in img_labels:
                    if len(label) >= 9:
                        _, x1, y1, x2, y2, x3, y3, x4, y4 = label[:9]
                        x1_pixel, y1_pixel = int(x1 * width), int(y1 * height)
                        x2_pixel, y2_pixel = int(x2 * width), int(y2 * height)
                        x3_pixel, y3_pixel = int(x3 * width), int(y3 * height)
                        x4_pixel, y4_pixel = int(x4 * width), int(y4 * height)

                        plt.gca().add_patch(plt.Polygon(
                            [(x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (x3_pixel, y3_pixel), (x4_pixel, y4_pixel)],
                            edgecolor='red', facecolor='none', lw=2
                        ))

                plt.title(f"Image: {img_name}")
            else:
                plt.title(f"Image: {img_name} (No Labels Available)")

            plt.axis('off')
            plt.show()