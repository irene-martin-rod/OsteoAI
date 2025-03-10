import os
import sys
sys.path.append(os.path.abspath('../scr'))
from import_images import create_image_dataset
from image_preprocesser import preprocess_image, apply_preprocessing
from extract_features import extract_features
import tensorflow as tf
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV



#Directories 
train_dir = "../Data/Processed/ml-dp/train"
val_dir = "../Data/Processed/ml-dp/val"
test_dir = "../Data/Processed/ml-dp/test"


train_ds = create_image_dataset(train_dir, subset = None, validation_split = None, image_size = (224, 224))
val_ds = create_image_dataset(val_dir, subset = None, validation_split = None, image_size = (224, 224))
test_ds = create_image_dataset(test_dir, subset = None, validation_split = None, image_size = (224, 224))
class_names = train_ds.class_names


#Preprocessing images
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
    rescale=1./255 
)



conv_base_VGG16 = VGG16(weights='imagenet', #Charging the pre-trined weights of VGG16 models with  ImageNet dataset
                  include_top=False, #This option indicates that the last layers (fully connected layers) used for final classification don't be imported
                  input_shape=(224, 224, 3)) #This CNN is optimized with a 224x224 resolution

conv_base_VGG16.trainable = False #freezing layers

conv_base_VGG16.summary()




#Unfreezing the lats 8 layers
for layer in conv_base_VGG16.layers[-8:]:
    layer.trainable = True


#Extract features
train_features, train_labels = extract_features(conv_base_VGG16, train_ds)
val_features, val_labels = extract_features(conv_base_VGG16, val_ds)
test_features, test_labels = extract_features(conv_base_VGG16, test_ds)


#Flatering features to vonverto to a 2D array for machine learning algorithms
train_features = train_features.reshape(train_features.shape[0], -1) #[0] number of samples in the dataset and -1 indicated to Numpy to 
#automatically calculate the size of the second dimension to preserve the total number of elements.
val_features = val_features.reshape(val_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)


num_classes = len(np.unique(train_labels))
params = [{
    "booster": ["gbtree"],
    "eta": [0.1, 0.05], 
    "gamma": [0.1, 0.5],  
    "max_depth": [3, 4],  
    "min_child_weight": [100, 200],  
    "subsample": [0.8, 1.0],  
    "colsample_bytree": [0.8, 0.9],  
    "lambda": [0.01, 0.1],  
    "alpha": [0.01, 0.1],  
    "tree_method": ["hist"],  
    "objective": ["multi:softmax"]  
}]

xgb = RandomizedSearchCV(XGBClassifier(objective="multi:softmax", num_class=num_classes), params, cv = 5, n_iter = 100, verbose = 1, n_jobs = 2)
xgb.fit(train_features, np.ravel(train_labels))

print("")
print("Best estimator found by random search:")
print(xgb.best_estimator_)