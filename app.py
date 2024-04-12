import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
import warnings
import base64


@st.cache_resource
def load_keras_model(model_path):
    model = load_model(model_path)
    return model
# Load the saved models
model_1 = load_keras_model('Classfication_model.h5')
model_2 = load_keras_model('Classfication_B_V_model.h5')

# Define class labels
class_labels_1 = ["NORMAL", "PNEUMONIA"]
class_labels_2 = ["BACTERIAL", "VIRAL"]

# Function to preprocess the input image
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    if img.shape[-1] != 3:
        img = np.stack((img,)*3, axis=-1)  # Convert grayscale to RGB
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to make predictions
def predict_image(model, image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction

# Function to get predicted class and probability
def get_predicted_class(prediction, class_labels):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    probability = prediction[0][predicted_class_index]
    return predicted_class, probability

# Streamlit app
st.title('Pneumonia X-Ray Detection')

uploaded_file = st.file_uploader("Choose an x-ray image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict using model 1
    prediction_1 = predict_image(model_1, image)
    predicted_class_1, probability_1 = get_predicted_class(prediction_1, class_labels_1)

    st.success(f"I am {probability_1*100:.2f}% sure that this is a {predicted_class_1} case.")

    # If pneumonia, ask to check type
    if predicted_class_1 == "PNEUMONIA":
        if st.button('Check if bacterial or viral'):
            # Predict using model 2
            prediction_2 = predict_image(model_2, image)
            predicted_class_2, probability_2 = get_predicted_class(prediction_2, class_labels_2)

            st.success(f"The pneumonia is {predicted_class_2} with a confidence of {probability_2*100:.2f}%.")



