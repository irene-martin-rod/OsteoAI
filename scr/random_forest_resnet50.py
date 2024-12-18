import os
import sys
sys.path.append(os.path.abspath('../scr'))
from import_images import create_image_dataset
from image_loader import LoadImage
from image_preprocesser import preprocess_image, apply_preprocessing
from callbacks_training_CNN import create_reduce_lr_callback, create_early_stopping_callback, establish_checkpoints , train_model
from metrics_CNN import plot_training_history, plot_confusion_matrix, plot_auc_curve
from extract_features import extract_features
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight




#Directories 
train_dir = "../Data/Processed/ml-dp/train"
val_dir = "../Data/Processed/ml-dp/val"
test_dir = "../Data/Processed/ml-dp/test"



#Load ResNet
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Freezing the layers
resnet_base.trainable = False

resnet_base.summary()



train_ds = create_image_dataset(train_dir, subset = None, validation_split = None, image_size=(224, 224), batch_size=32)
val_ds = create_image_dataset(val_dir, subset = None, validation_split = None, image_size=(224, 224), batch_size=32)
test_ds = create_image_dataset(test_dir, subset = None, validation_split = None, image_size=(224, 224), batch_size=32)
class_names = train_ds.class_names


train_ds = apply_preprocessing(
    train_ds,
    rescale=1./255 
)

val_ds = apply_preprocessing(
    val_ds,
    rescale=1./255 
)

test_ds = apply_preprocessing(
    test_ds,
    rescale=1./255)




#Unfreezing the lats 8 layers
for layer in resnet_base.layers[-12:]:
    layer.trainable = True



#Extract features
train_features, train_labels = extract_features(resnet_base, train_ds)
val_features, val_labels = extract_features(resnet_base, val_ds)
test_features, test_labels = extract_features(resnet_base, test_ds)


#Flatering features to vonverto to a 2D array for machine learning algorithms
train_features = train_features.reshape(train_features.shape[0], -1) #[0] number of samples in the dataset and -1 indicated to Numpy to 
#automatically calculate the size of the second dimension to preserve the total number of elements.
val_features = val_features.reshape(val_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)


#Training the Machine Learning Model

params = [{
    "n_estimators": [100, 500, 1000, 1500],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": range(1, 100),
    "min_samples_split": range(100, 501)
}]

#Random forest
random_forest = RandomizedSearchCV(RandomForestClassifier(), params, cv = 5, n_iter = 1000, verbose = 1, random_state=42, n_jobs = 6)
random_forest.fit(train_features, np.ravel(train_labels))

print("")
print("Best estimator found by random search:")
print(random_forest.best_estimator_)