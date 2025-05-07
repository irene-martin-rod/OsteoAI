# === IMPORTS ===
import streamlit as st
from PIL import Image
import os
import sys
import base64
sys.path.append(os.path.abspath('../src'))
from image_preprocesser import preprocess_image
from extract_features import extract_features
import tensorflow as tf
from keras.applications import VGG16
import numpy as np
import joblib

# === PAGE SETTINGS ===
st.set_page_config(page_title="OsteoAI", page_icon="🦴", layout="wide")

# === CUSTOM STYLES ===
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #e3f2fd;
        }
        .title {
            color: #1e88e5;
            font-size: 3rem;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #555;
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 2rem;
        }
        .card {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem auto;
            width: 300px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .fracture {
            color: #e57373;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .nofracture {
            color: #81c784;
            font-weight: bold;
            font-size: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<div class='title'>🦴 OsteoAI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Automatic X-ray Fracture Classifier</div>", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    #st.image("logo.png", use_container_width=True)
    st.markdown("## 📤 Upload X-rays")
    uploaded_files = st.file_uploader("Choose JPG/PNG files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
        OsteoAI helps detect bone fractures from X-rays using deep learning.
        \nModel: VGG16 + LightGBM
    """)
    st.markdown("⚠️ This is a theoretical prototype. Not for medical use.")

# === MODEL LOADING ===
@st.cache_resource
def load_model():
    return joblib.load("lgbm.pkl")

@st.cache_resource
def load_vgg16():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-8:]:
        layer.trainable = True
    return base_model

model = load_model()
vgg16 = load_vgg16()

# === DEMO EXAMPLES ===
st.markdown("### 🧪 Examples")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🟥 Fracture Example")
    image = Image.open("example_fracture.jpg").resize((300, 300))
    st.image(image, caption="Fracture Detected", use_container_width=False)
    st.markdown("<p class='fracture'>🟥 Fracture</p>", unsafe_allow_html=True)

with col2:
    st.markdown("#### 🟩 No Fracture Example")
    image = Image.open("example_nofracture.jpg").resize((300, 300))
    st.image(image, caption="Healthy X-ray", use_container_width=False)
    st.markdown("<p class='nofracture'>🟩 No Fracture</p>", unsafe_allow_html=True)

st.markdown("---")

# === PREDICTION ===
if uploaded_files:
    st.markdown("### 🔍 Results")

    images_to_predict = []
    image_data = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB").resize((224, 224))
        image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
        preprocessed, _ = preprocess_image(image_tensor, label=0, rescale=1./255, output_size=(224, 224))
        images_to_predict.append(preprocessed)

        # Base64 encode
        buffered = image.copy()
        buffered.save("temp.png", format="PNG")
        with open("temp.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        image_data.append(encoded)

    with st.spinner("Analyzing..."):
        batch_tensor = tf.stack(images_to_predict)
        features = vgg16.predict(batch_tensor)
        features_flat = features.reshape((features.shape[0], -1))
        predictions = model.predict(features_flat)

    # SHOW PREDICTIONS
    for i, pred in enumerate(predictions):
        label = "🟩 No Fracture" if pred == 1 else "🟥 Fracture"
        css = "nofracture" if pred == 1 else "fracture"

        st.markdown(f"""
            <div class='card'>
                <img src='data:image/png;base64,{image_data[i]}' style='width:100%; border-radius:10px;'/>
                <p class='{css}'>{label}</p>
            </div>
        """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#999;'>© 2025 OsteoAI</div>", unsafe_allow_html=True)
