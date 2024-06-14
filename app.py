import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
MODEL_PATH = 'sugarcane3.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class names for your model
class_names = ['RedRot', 'RedRust']

# Function to preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions using the trained model
def predict(image, model):
    img_width, img_height = model.input_shape[1:3]
    image = preprocess_image(image, (img_width, img_height))
    prediction = model.predict(image)
    return prediction

def main():
    st.set_page_config(page_title="Sugarcane  Disease Detection", layout="wide")

    st.markdown("""
        <style>
        .main {
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url(https://images.unsplash.com/photo-1536882240095-0379873feb4e?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1771&q=80);
            background-size: cover;
            background-position: center;
            color: #fff;
            padding: 50px 0;
        }
        .welcome-text h1 {
            color: #ffe15d;
            font-size: 60px;
            text-shadow: 3px 3px black;
            text-align: center;
        }
        .file-upload {
            text-align: center;
            margin-top: 30px;
        }
        .file-upload label {
            display: inline-block;
            color: #ffffff;
            border: 1px solid #fff;
            font-size: 18px;
            padding: 10px 20px;
            cursor: pointer;
            margin-top: 20px;
        }
        .file-upload label:hover {
            background: #fff;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main">
            <div class="welcome-text">
                <h1>Sugarcane  Disease Detection</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="file-upload">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False, width=300)  # Set the width to 300 pixels

        prediction = predict(image, model)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]

        st.markdown("""
            <div class="welcome-text">
                <h1>The Sugarcane  is classified as: <span style="color:#ffe15d;">{}</span></h1>
            </div>
        """.format(predicted_class_name), unsafe_allow_html=True)

if __name__ == "__main__":
    main()