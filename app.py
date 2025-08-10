import streamlit as st
import pandas as pd
import pickle

# Load trained logistic regression model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page title
st.title("🏦 Loan Status Prediction App")
st.write("Enter the details below to check if the loan will be approved or not.")

# User input form
with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", [1, 0])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])

    submitted = st.form_submit_button("Predict Loan Status")

# Encoding mappings (must match your training LabelEncoder mapping)
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
education_map = {"Graduate": 0, "Not Graduate": 1}
self_employed_map = {"Yes": 1, "No": 0}
property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# Make prediction after form submission
if submitted:
    # Prepare input data
    input_data = pd.DataFrame([{
        'Gender': gender_map[gender],
        'Married': married_map[married],
        'Education': education_map[education],
        'Self_Employed': self_employed_map[self_employed],
        'Property_Area': property_map[property_area],
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'LoanAmountTerm': loan_amount_term,
        'Credit_History': credit_history,
        'Dependents': dependents
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("Loan is likely to be Approved!")
    else:
        st.error("Loan is likely to be Rejected.")
