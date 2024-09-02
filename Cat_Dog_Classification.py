# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:54:17 2024

@author: prachet
"""

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import MyModel  

# Load your trained model
model = MyModel()
model.load_state_dict(torch.load(r"C:\Users\prachet\OneDrive - Vidyalankar Institute of Technology\Desktop\Coding\Machine Learning\DEEP LEARNING\DL Project 3. Dog vs Cat Classification using Transfer Learning\cat_dog_classification_trained_model.sav", 'rb'))
model.eval()

# Define transformations to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image):
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output).item()

    return prediction

# Streamlit UI
st.title("Cat or Dog Classifier")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make prediction
    prediction = predict_image(image)
    if prediction == 0:
        st.write("Prediction: It's a cat!")
    else:
        st.write("Prediction: It's a dog!")
