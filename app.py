import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Keras model
@st.cache_resource
def load_keras_model():
    return load_model("model.keras")

model = load_keras_model()

# Streamlit app title
st.markdown('<p style="font-size:22px; font-weight:bold;">Heart Disease Prediction App</p>', unsafe_allow_html=True)

# Create input fields for user input with descriptions
# st.write("Please fill in the details below to predict the likelihood of heart disease.")

age = st.slider("Age", min_value=18, max_value=120, value=50, help="Age of the person in years.")

sex = st.selectbox("Sex", options=[0, 1], index=0, help="Select the sex: 0 for Female, 1 for Male.")
# st.write("Sex Encoding: 0 = Female, 1 = Male")

chest_pain = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], index=0, help="Chest pain types: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic.")
# st.write("Chest Pain Encoding: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic")

resting_bp = st.slider("Resting Blood Pressure", min_value=80, max_value=200, value=120, help="Resting blood pressure in mm Hg.")
cholesterol = st.slider("Cholesterol", min_value=100, max_value=600, value=200, help="Cholesterol level in mg/dl.")
fasting_bs = st.selectbox("Fasting Blood Sugar", options=[0, 1], index=0, help="Fasting blood sugar: 0 = < 120 mg/dl, 1 = > 120 mg/dl.")
# st.write("Fasting Blood Sugar Encoding: 0 = < 120 mg/dl, 1 = > 120 mg/dl")

resting_ecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], index=0, help="Resting ECG results: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.")
# st.write("Resting ECG Encoding: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy")

max_hr = st.slider("Max Heart Rate", min_value=50, max_value=220, value=150, help="Maximum heart rate achieved.")
exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], index=0, help="Exercise induced angina: 0 = No, 1 = Yes.")
# st.write("Exercise Induced Angina Encoding: 0 = No, 1 = Yes")

old_peak = st.slider("Oldpeak", min_value=-2.0, max_value=6.0, value=1.0, help="Depression induced by exercise relative to rest.")
st_slope = st.selectbox("ST Slope", options=[0, 1, 2], index=0, help="ST slope: 0 = Upsloping, 1 = Flat, 2 = Downsloping.")
# st.write("ST Slope Encoding: 0 = Upsloping, 1 = Flat, 2 = Downsloping")



# Create input array
input_features = np.array([[
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    old_peak,
    st_slope
]])

# Predict button
if st.button("Heart Disease Prediction: "):
    prediction = model.predict(input_features)
    predicted_class = int(prediction[0][0] > 0.5)  # Assuming binary classification (0 or 1)
    
    # Show prediction
    if predicted_class == 1:
        st.success("The model predicts: High Risk of Heart Disease (1)")
    else:
        st.success("The model predicts: Low Risk of Heart Disease (0)")

