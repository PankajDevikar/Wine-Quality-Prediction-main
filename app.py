import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Define paths for model and image
model_path = r'C:\data science material\wine quality prediction\my_streamlit_app\Notebook\random_forest_model.joblib'
image_path = 'C:/data science material/wine quality prediction/my_streamlit_app/images/wine pic.jpeg'

# Load the model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}. Please ensure the file exists.")
    st.stop()

# Load and display the wine image
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption='Wine Quality Prediction', use_column_width=True)
else:
    st.warning(f"Image file not found at {image_path}. Proceeding without displaying the image.")

# Title of the app
st.title("Wine Quality Prediction App")

# Subtitle
st.write("Enter the wine features below to predict its quality:")

# Input fields for wine features
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, value=7.4)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, value=0.5)
citric_acid = st.number_input('Citric Acid', min_value=0.0, value=0.36)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, value=6.1)
chlorides = st.number_input('Chlorides', min_value=0.0, value=0.071)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, value=17.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, value=102.0)
density = st.number_input('Density', min_value=0.0, value=0.9978)
pH = st.number_input('pH', min_value=0.0, value=2.35)
sulphates = st.number_input('Sulphates', min_value=0.0, value=0.8)
alcohol = st.number_input('Alcohol', min_value=0.0, value=10.5)

# Predict button
if st.button("Predict"):
    # Prepare the input data as a NumPy array
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    
    # Predict the wine quality
    try:
        prediction = model.predict(input_data)
        
        # Display the result
        if prediction[0] >= 6:  # Adjusting the threshold to define "Good" quality (e.g., quality >= 6)
            st.success("This is a Good Quality Wine!")
        else:
            st.error("This is a Bad Quality Wine!")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.write("Wine quality is predicted using a Random Forest model.")


