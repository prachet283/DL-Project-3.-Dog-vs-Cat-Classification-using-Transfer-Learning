# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:06:31 2024

@author: prachet
"""
import numpy as np
import pickle
import streamlit as st
import cv2
from PIL import Image
# Load your trained model


model = pickle.load(open(r"C:\Users\prachet\OneDrive - Vidyalankar Institute of Technology\Desktop\Coding\Machine Learning\DEEP LEARNING\DL Project 3. Dog vs Cat Classification using Transfer Learning\cat_dog_classification_trained_model.sav")) 

def predict_image(image):
    # Preprocess the image
    input_img_resize = cv2.resize(image, (224,224))
    input_img_scaled = input_img_resize/255
    image_reshaped = np.reshape(input_img_scaled, [1,224,224,3])
    input_prediction = model.predict(image_reshaped)
    print(input_prediction)

    input_pred_label = np.argmax(input_prediction)
    print(input_pred_label)
    return input_pred_label



def main():
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

if __name__ == '__main__':
    main()