import streamlit as st
import pickle
import numpy as np
import os

model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'random_forest_classifier_model.pkl'), 'rb'))

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=0)

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thal (0-3)", options=[0, 1, 2, 3])

if st.button("🔍 Predict"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("⚠️ The person IS likely to have Heart Disease!")
    else:
        st.success("✅ The person is NOT likely to have Heart Disease!")
