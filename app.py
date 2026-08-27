import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load saved models with error handling
models_loaded = False
try:
    with open(os.path.join(current_dir, "loan_model.pkl"), "rb") as file:
        model = pickle.load(file)

    with open(os.path.join(current_dir, "scaler.pkl"), "rb") as file:
        scaler = pickle.load(file)

    with open(os.path.join(current_dir, "encoder.pkl"), "rb") as file:
        encoder = pickle.load(file)
    
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.info("Please ensure loan_model.pkl, scaler.pkl, and encoder.pkl exist in the repository.")

# App title
st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below to check loan approval status.")
st.divider()

if models_loaded:
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        no_of_dependents = st.selectbox(
            "Number of Dependents",
            ["0", "1", "2", "3+"]
        )
        
        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )
        
        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )
    
    with col2:
        st.subheader("💰 Financial Information")
        income_annum = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            step=10000,
            value=0
        )
        
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0,
            step=10000,
            value=0
        )
        
        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=0,
            step=1,
            value=0
        )
    
    st.subheader("📊 Assets & Credit Score")
    col3, col4 = st.columns(2)
    
    with col3:
        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=0,
            max_value=900,
            step=10,
            value=0
        )
        
        residential_assets_value = st.number_input(
            "Residential Assets Value (₹)",
            min_value=0,
            step=10000,
            value=0
        )
    
    with col4:
        commercial_assets_value = st.number_input(
            "Commercial Assets Value (₹)",
            min_value=0,
            step=10000,
            value=0
        )
        
        luxury_assets_value = st.number_input(
            "Luxury Assets Value (₹)",
            min_value=0,
            step=10000,
            value=0
        )
    
    bank_asset_value = st.number_input(
        "Bank Asset Value (₹)",
        min_value=0,
        step=10000,
        value=0
    )
    
    st.divider()
    
    # Prediction button
    if st.button("🔍 Check Loan Approval", use_container_width=True):
        
        try:
            # Create numerical data dataframe matching main.py column order and types
            numeric_data = pd.DataFrame([{
                "income_annum": float(income_annum),
                "loan_amount": float(loan_amount),
                "loan_term": float(loan_term),
                "cibil_score": float(cibil_score),
                "residential_assets_value": float(residential_assets_value),
                "commercial_assets_value": float(commercial_assets_value),
                "luxury_assets_value": float(luxury_assets_value),
                "bank_asset_value": float(bank_asset_value)
            }])
            
            # Create categorical data dataframe - ensure values match training data format exactly
            categorical_data = pd.DataFrame([{
                "no_of_dependents": str(no_of_dependents),
                "education": str(education),
                "self_employed": str(self_employed)
            }])
            
            # Scale numerical data using the fitted scaler
            numeric_scaled = scaler.transform(numeric_data)
            
            # Encode categorical data using the fitted encoder
            categorical_encoded = encoder.transform(categorical_data)
            
            # Combine scaled numerical and encoded categorical features
            final_data = np.hstack([
                numeric_scaled,
                categorical_encoded
            ])
            
            # Make prediction
            prediction = model.predict(final_data)[0]
            prediction_proba = model.predict_proba(final_data)[0]
            
            # Display results
            st.divider()
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:
                st.success("✅ LOAN APPROVED!", icon="✅")
                approval_prob = prediction_proba[1] * 100
                st.info(f"Approval Confidence: {approval_prob:.2f}%")
            else:
                st.error("❌ LOAN REJECTED!", icon="❌")
                rejection_prob = prediction_proba[0] * 100
                st.warning(f"Rejection Probability: {rejection_prob:.2f}%")
            
            # Display detailed input summary
            with st.expander("📋 View Input Summary"):
                col_summary1, col_summary2 = st.columns(2)
                
                with col_summary1:
                    st.write("**Personal Information:**")
                    st.write(f"• Dependents: {no_of_dependents}")
                    st.write(f"• Education: {education}")
                    st.write(f"• Self Employed: {self_employed}")
                    
                    st.write("**Financial Information:**")
                    st.write(f"• Annual Income: ₹{income_annum:,.0f}")
                    st.write(f"• Loan Amount: ₹{loan_amount:,.0f}")
                    st.write(f"• Loan Term: {loan_term} months")
                
                with col_summary2:
                    st.write("**Assets & Credit:**")
                    st.write(f"• CIBIL Score: {cibil_score}")
                    st.write(f"• Residential Assets: ₹{residential_assets_value:,.0f}")
                    st.write(f"• Commercial Assets: ₹{commercial_assets_value:,.0f}")
                    st.write(f"• Luxury Assets: ₹{luxury_assets_value:,.0f}")
                    st.write(f"• Bank Assets: ₹{bank_asset_value:,.0f}")
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Please check your inputs and try again.")

else:
    st.warning("⚠️ Models not loaded successfully. Please contact support.")
