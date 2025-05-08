# === IMPORTS ===
import streamlit as st
from PIL import Image
import os
import sys
import base64
import numpy as np
import tensorflow as tf
import joblib
from keras.applications import VGG16
import base64

sys.path.append(os.path.abspath('../src'))
from image_preprocesser import preprocess_image
from extract_features import extract_features

# === PAGE SETTINGS ===
st.set_page_config(page_title="OsteoAI", page_icon="🦴", layout="wide")

# === LOAD FONTS ===
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# === CUSTOM CSS ===
st.markdown("""
    <style>
        body {
            background-color: #e3f2fd;
        }
        div.block-container {
            background-color: #e3f2fd !important;
            padding-top: 2rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #C4E9F7 !important;
        }

        .title {
            color: #0a4475;
            font-size: 3rem;
            text-align: center;
            margin-bottom: 0.2rem;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
        }

        .subtitle {
            color: #0a4475;
            text-align: center;
            font-size: 1.5rem;
            margin-bottom: 2rem;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
        }
            
        .section-title {
            font-family: 'Montserrat', sans-serif;
            color: #0A4475;
            font-weight: bold;
            font-size: 1.05rem;
            margin-top: 5rem;
            margin-bottom: 1rem;
}

        .card {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem auto;
            width: 400px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        .fracture-box {
            background-color: #e57373;
            color: white;
            font-weight: bold;
            font-size: 1rem;
            padding: 0.5rem;
            border-radius: 10px;
            margin-top: 10px;
        }

        .nofracture-box {
            background-color: #81c784;
            color: white;
            font-weight: bold;
            font-size: 1rem;
            padding: 0.5rem;
            border-radius: 10px;
            margin-top: 10px;
        }

        .upload-area {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-top: 1rem;
        }

        .upload-icon {
            font-size: 5rem;
            color: #1e88e5;
        }

        .about-section {
            margin-top: 2rem; 
            text-align: left;
            color: #0A4475;
        }
            
        .about-section-title {
            font-size: 1.05rem;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
            margin-bottom: 1rem; 
        }

        .about-section p {
            font-size: 1rem;
            text-align: justify;
        }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<div class='title'>OsteoAI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Automatic X-Ray Classifier for Bone Fractures by Artificial Intelligence</div>", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    logo_encoded = base64.b64encode(open("logo.png", "rb").read()).decode()

    # CSS to personilze the uploader
    st.markdown(f"""
        <style>
        /* Logo box (without upper margin) */
        .logo-container {{
            text-align: center;
            margin-top: 0rem; /* No margin */
            margin-bottom: 2rem; /* Space below  */
        }}

        .logo-container img {{
            width: 150px; /* Logo size */
        }}

        /* Text "Upload your X-ray images" */
        .upload-text {{
            color: #0A4475;
            font-weight: bold;
            font-size: 2rem; 
            margin-top: 2rem;
            font-family: 'Montserrat', sans-serif;
        }}

        /* Uploader box */
        section[data-testid="stFileUploader"] {{
            background-color: #f5f5f5;
            border-radius: 12px;
            padding: 2rem 1rem;
            color: #0A4475;
            text-align: center;
            margin-top: 3rem;
            margin-bottom: 7rem;
        }}

        /* Inner style of uploader box */
        section[data-testid="stFileUploader"] > div {{
            border: 2px dashed #0A4475;
            border-radius: 8px;
            background-color: #ffffff;
            padding: 1.2rem;
            color: #000000;
            font-weight: bold;
        }}

        /* Hide automatic label */
        section[data-testid="stFileUploader"] label {{
            display: none !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Logo ans text insided box
    st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_encoded}" />
        </div>
        <p class="upload-text">Upload your X-ray images</p>
    """, unsafe_allow_html=True)

    # Uploader
    uploaded_files = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
        <div class="about-section">
        <div class="about-section-title">About</div>
        <p>OsteoAi is an application that uses deep learning and machine learning models to classify X-ray images and detect potential bone fractures.</p>
        <p>⚠️ This is a theoretical prototype. Not for medical use.</p>
        <p>Author: Irene Martín Rodríguez</p>
        <p>Last updated: May 2025</p>
        <p>MIT License</p>
    </div>
    """, unsafe_allow_html=True)



# === EXAMPLES ===
st.markdown("<div class='section-title'>Examples</div>", unsafe_allow_html=True)

@st.cache_data
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

fracture_img_base64 = image_to_base64("example_fracture.jpg")
nofracture_img_base64 = image_to_base64("example_nofracture.jpg")


col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
        <div class='card'>
            <img src='data:image/jpeg;base64,{fracture_img_base64}' style='width:100%; height:400px; object-fit: cover; border-radius:10px; margin-bottom:10px;'/>
            <div class='fracture-box'>Fracture</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='card'>
            <img src='data:image/jpeg;base64,{nofracture_img_base64}' style='width:100%; height:400px; object-fit: cover; border-radius:10px; margin-bottom:10px;'/>
            <div class='nofracture-box'>No Fracture</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

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


# === PREDICTION ===
if uploaded_files:
    st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)

    images_to_predict = []
    image_data = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB").resize((224, 224))
        image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
        preprocessed, _ = preprocess_image(image_tensor, label=0, rescale=1./255, output_size=(224, 224))
        images_to_predict.append(preprocessed)

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

    cols = st.columns(len(predictions))
    for i, pred in enumerate(predictions):
        label = "No Fracture" if pred == 1 else "Fracture"
        css_class = "nofracture-box" if pred == 1 else "fracture-box"

        with cols[i]:
            st.markdown(f"""
                <div class='card'>
                    <img src='data:image/png;base64,{image_data[i]}' style='width:100%; border-radius:10px;'/>
                    <div class='{css_class}'>{label}</div>
                </div>
            """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#999;'>© 2025 OsteoAI</div>", unsafe_allow_html=True)
