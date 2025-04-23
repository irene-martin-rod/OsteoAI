# Script with the model workflow used for model deploy in Streamlit


#Import libraries
import os
import sys
sys.path.append(os.path.abspath('../src'))
from import_images import create_image_dataset
from image_preprocesser import preprocess_image, apply_preprocessing
from extract_features import extract_features
import tensorflow as tf
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
import joblib

#Import model
lgb = joblib.load("../App/lgbm.pkl")

#Import image and preprocessing
