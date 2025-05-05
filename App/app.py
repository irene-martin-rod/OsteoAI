# Script with the model workflow used for model deploy in Streamlit


#Import libraries
import streamlit as st
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath('../src'))
from image_preprocesser import preprocess_image
from extract_features import extract_features
import tensorflow as tf
from keras.applications import VGG16
import numpy as np
import joblib


# === APP SETTINGS ===
st.set_page_config(page_title="🦴 OsteoAI: Fracture Classifier", layout="centered")
st.title("Automatically AI classificator of X-Ray Images")




# === MODEL LOADING ===¨
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


model = load_model()
vgg16 = load_vgg16()


# === IMAGE UPLOADER & PREDICTION ===
uploaded_files = st.file_uploader("Upload X-ray images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### Uploaded Images")
    images_to_predict = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"{file.name}", use_column_width=True)

        # Convert and preprocess image
        image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
        image_preprocessed, _ = preprocess_image(image_tensor, label=0, rescale=1./255, output_size=(224, 224))
        images_to_predict.append(image_preprocessed)

    # === BATCH PREDICTION ===
    batch_tensor = tf.stack(images_to_predict)
    features = vgg16.predict(batch_tensor)
    features_flat = features.reshape((features.shape[0], -1))
    predictions = model.predict(features_flat)

    st.markdown("### 🧠 Predictions:")
    for i, pred in enumerate(predictions):
        label = "🟥 Fracture" if pred == 0 else "🟩 No Fracture"
        st.write(f"**{uploaded_files[i].name}** → {label}")




