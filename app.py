import streamlit as st
import pandas as pd
import numpy as np
import pickle


# --------------------------------------------------
# Load saved model, scaler and encoder
# --------------------------------------------------

with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦"
)

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below.")


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

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
    min_value=0,
    step=1
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    step=1
)

loan_term = st.number_input(
    "Loan Term",
    min_value=0,
    step=1
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=0,
    max_value=900,
    step=1
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    step=1
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    step=1
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    step=1
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    step=1
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🔍 Check Loan Approval"):

    # --------------------------------------------------
    # Numerical data
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Categorical data
    # --------------------------------------------------

    categorical_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed
    }])


    # --------------------------------------------------
    # Scaling numerical features
    # --------------------------------------------------

    numeric_scaled = scaler.transform(numeric_data)


    # --------------------------------------------------
    # Encoding categorical features
    # --------------------------------------------------

    categorical_encoded = encoder.transform(categorical_data)


    # --------------------------------------------------
    # Combine features
    # IMPORTANT:
    # Same order as training code
    # Numeric features FIRST
    # Categorical features SECOND
    # --------------------------------------------------

    final_data = np.hstack([
        numeric_scaled,
        categorical_encoded
    ])


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(final_data)[0]


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected!")
