import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page config
st.set_page_config(page_title="Dog Breed Classifier", layout="centered")

# Title
st.title("🐶 Dog Breed Classification App")
st.write("Upload a dog image to predict its breed")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("dog_breed_model.keras")
    return model

model = load_model()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# File uploader
uploaded_file = st.file_uploader("Upload Dog Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (IMPORTANT for MobileNetV2)
    img = image.resize((160, 160))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_breed = class_names[predicted_index]

    # Result
    st.subheader("Prediction Result:")
    st.success(f"Predicted Breed: {predicted_breed}")
    st.info(f"Confidence: {confidence * 100:.2f}%")