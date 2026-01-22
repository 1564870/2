import streamlit as st
import joblib
import numpy as np
import math

# Load the model
model = joblib.load('SVM_label.pkl')

# Feature names
feature_names = [
    'ProtrudingSegments',
    'Osteoporosis',
    'Diabetes',
    'Hypertension',
    'ResNet101_transverse_DTL1346',
    'ResNet101_transverse_DTL431',
    'ResNet101_sagittal_DTL552',
    'ResNet101_transverse_DTL653',
    'ResNet101_transverse_DTL648',
    'ResNet101_transverse_DTL1078',
    'ResNet101_sagittal_DTL459',
    'ResNet101_sagittal_DTL434',
    'ResNet101_sagittal_DTL1563',
    'ResNet101_sagittal_DTL348',
    'ResNet101_sagittal_DTL1066',
    'ResNet101_transverse_DTL565',
    'ResNet101_sagittal_DTL1671',
    'ResNet101_transverse_DTL1053',
    'ResNet101_transverse_DTL1222',
    'ResNet101_transverse_DTL1409',
    'ResNet101_sagittal_DTL947',
    'ResNet101_transverse_DTL1994',
    'ResNet101_sagittal_DTL1167',
    'ResNet101_transverse_DTL262',
    'ResNet101_transverse_DTL778',
    'ResNet101_transverse_DTL2010',
    'ResNet101_transverse_DTL1653',
    'ResNet101_sagittal_DTL381',
    'ResNet101_transverse_DTL38'
]

# Title and description
st.title("Post-Surgery Sensory Improvement Predictor")
st.write("Enter the following details to predict post-surgery numbness improvement.")

# Input fields
input_data = {}

# Dropdowns for categorical features
input_data['Osteoporosis'] = st.selectbox('Osteoporosis (0=No, 1=Yes):', options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
input_data['Hypertension'] = st.selectbox('Hypertension (0=No, 1=Yes):', options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
input_data['Diabetes'] = st.selectbox('Diabetes (0=No, 1=Yes):', options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
input_data['ProtrudingSegments'] = st.selectbox('ProtrudingSegments:', options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}[x])

# Numeric input fields for other features
for feature in feature_names:
    if feature not in ["Osteoporosis", "Hypertension", "Diabetes", "ProtrudingSegments"]:
        input_data[feature] = st.number_input(f"{feature}:", value=0.0)

# Prediction button
if st.button("Predict"):
    # Convert inputs to a format suitable for the model
    try:
        feature_values = [input_data[feature] for feature in feature_names]
        features = np.array([feature_values], dtype=float)

        # Make prediction
        predicted_class = int(model.predict(features)[0])

        # Predict probabilities (with fallback if predict_proba is unavailable)
        predicted_proba = None
        if hasattr(model, "predict_proba"):
            predicted_proba = model.predict_proba(features)[0]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(features)
            score = float(score[0]) if np.ndim(score) > 0 else float(score)
            p1 = 1.0 / (1.0 + math.exp(-score))
            predicted_proba = np.array([1.0 - p1, p1], dtype=float)

        # Generate prediction results and advice
        result = f"Predicted Class: {predicted_class}\n"
        if predicted_proba is not None:
            result += f"Prediction Probabilities: {predicted_proba}\n"
        else:
            result += "Prediction Probabilities: not available for this model.\n"

        advice = ""

        if predicted_proba is not None and predicted_class < len(predicted_proba):
            probability = predicted_proba[predicted_class] * 100
            if predicted_class == 1:
                advice = (
                    f"\nBased on our model's prediction, your numbness improvement after the surgery is significant. "
                    f"The model predicts a {probability:.1f}% likelihood of significant postoperative improvement in numbness."
                )
            else:
                advice = (
                    f"\nBased on our model's prediction, your numbness improvement after the surgery is limited. "
                    f"The model predicts a {probability:.1f}% likelihood of limited improvement."
                )
        else:
            if predicted_class == 1:
                advice = (
                    f"\nBased on our model's prediction, your numbness improvement after the surgery is significant."
                )
            else:
                advice = (
                    f"\nBased on our model's prediction, your numbness improvement after the surgery is limited."
                )

        # Display result and advice
        st.success(result + advice)

    except Exception as e:
        st.error(f"An error occurred: {e}")
