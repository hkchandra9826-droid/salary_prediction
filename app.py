import streamlit as st
import joblib
import pandas as pd

# Load trained model and encoder
model = joblib.load("salary_model.pkl")
encoder = joblib.load("education_encoder.pkl")

# Website title
st.title("Salary Prediction")

st.write("Enter your details to predict your salary.")

# Education input
education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

# Experience input
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0
)

# Prediction
if st.button("Predict Salary"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Education Level": [education],
        "Years of Experience": [experience]
    })

    # Encode education level
    input_data[["Education Level"]] = encoder.transform(
        input_data[["Education Level"]]
    )

    # Predict salary
    prediction = model.predict(input_data)

    # Display result
    st.success(
        f"Predicted Salary: ₹{prediction[0]:,.2f}"
    )