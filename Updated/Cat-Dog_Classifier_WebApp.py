import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_cat_dog_model.h5")
    return model

model = load_model()
IMG_SIZE = 224

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and find out if it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"Confidence: `{confidence:.8f}`")
