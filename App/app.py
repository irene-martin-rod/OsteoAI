# Script with the model workflow used for model deploy in Streamlit

# === IMPORTS ===
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


# === PAGE SETTINGS ===
st.set_page_config(page_title="🦴 OsteoAI: Fracture Classifier", layout="centered")
st.markdown("""
    <style>
        .banner {
            background-color: #e3f2fd;
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            border: 2px solid #64b5f6;
            margin-bottom: 2rem;
        }
        .banner h1 {
            color: #1e88e5;
            font-size: 2.5rem;
        }
        .banner p {
            color: #333;
            font-size: 1.2rem;
        }

        .upload-box {
            border: 2px dashed #90caf9;
            padding: 2rem;
            text-align: center;
            border-radius: 10px;
            background-color: #f5faff;
            margin-bottom: 2rem;
        }

        .prediction-box {
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        .fracture {
            background-color: #ffebee;
            color: #c62828;
        }

        .nofracture {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
    </style>

    <div class="banner">
        <h1>🦴 OsteoAI</h1>
        <p>Automatic X-Ray Classifier for Bone Fractures</p>
    </div>
""", unsafe_allow_html=True)


# === MODEL LOADING ===
@st.cache_resource
def load_model():
    '''Function to load LGBM model'''
    return joblib.load("lgbm.pkl")

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


# === IMAGE UPLOADER ===
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("📤 Upload one or more X-ray images (JPG/PNG)", 
                                   type=["jpg", "jpeg", "png"], 
                                   accept_multiple_files=True)
st.markdown('</div>', unsafe_allow_html=True)


# === PREDICTION WORKFLOW ===
if uploaded_files:
    st.markdown("### 📸 Uploaded Images")
    images_to_predict = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=file.name, use_column_width=True)

        image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
        image_preprocessed, _ = preprocess_image(image_tensor, label=0, rescale=1./255, output_size=(224, 224))
        images_to_predict.append(image_preprocessed)

    with st.spinner("🧠 Processing images... please wait"):
        batch_tensor = tf.stack(images_to_predict)
        features = vgg16.predict(batch_tensor)
        features_flat = features.reshape((features.shape[0], -1))
        predictions = model.predict(features_flat)

    st.markdown("### 🔍 Predictions")
    for i, pred in enumerate(predictions):
        label_text = "🟩 No Fracture" if pred == 1 else "🟥 Fracture"
        css_class = "nofracture" if pred == 1 else "fracture"
        st.markdown(f"""
            <div class="prediction-box {css_class}">
                <strong>{uploaded_files[i].name}</strong><br>
                {label_text}
            </div>
        """, unsafe_allow_html=True)
