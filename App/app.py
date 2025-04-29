# Script with the model workflow used for model deploy in Streamlit


#Import libraries
import streamlit as st
from PIL import Image
from src.image_preprocesser import preprocess_image, apply_preprocessing
from src.extract_features import extract_features
import tensorflow as tf
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
import joblib


# === APP SETTINGS ===
st.set_page_config(page_title="🦴 OsteoAI: Fracture Classifier", layout="centered")
st.title("Automatically AI classificator of X-Ray Images")




# === LOAD MODEL AND PRETRAINED CNN ===
@st.cache_resource
def load_model():
    '''Function to load LBG model'''
    model = joblib.load("lgbm.pkl")
    return model

@st.cache_resource
def load_vgg16():
    '''Function to load VGG16 network'''
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    return base_model



# === LOAD MODELS ===
model = load_model()
vgg16 = load_vgg16()







