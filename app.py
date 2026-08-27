import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load saved files
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# App title
st.title("🏦 Loan Approval Prediction")

st.write("Enter the applicant details below.")


# User inputs
no_of_dependents = st.selectbox(
    "Number of Dependents",
    [0, 1, 2, 3, 4, 5]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=0
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=0,
    max_value=900
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0
)


# Prediction button
if st.button("Check Loan Approval"):

    # Numerical input
    numeric_data = pd.DataFrame([{
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    # Categorical input
    categorical_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed
    }])

    # Scale numerical data
    numeric_scaled = scaler.transform(numeric_data)

    # Encode categorical data
    categorical_encoded = encoder.transform(categorical_data)

    # Combine both
    final_data = np.hstack([
        numeric_scaled,
        categorical_encoded
    ])

    # Prediction
    prediction = model.predict(final_data)[0]

    # Display result
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected!")
