import streamlit as st
import pandas as pd
import pickle

# Load trained logistic regression model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature order that matches X.columns from training
FEATURE_ORDER = [
    'Gender', 'Married','Dependents', 'Education', 'Self_Employed', 'Property_Area',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm',
    'Credit_History'
]

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

input_df = None
prediction = None

# Make prediction after form submission
if submitted:
    input_dict = {
        'Gender': gender_map.get(gender, 0),
        'Married': married_map.get(married, 0),
        'Education': education_map.get(education, 0),
        'Self_Employed': self_employed_map.get(self_employed, 0),
        'Property_Area': property_map.get(property_area, 0),
        'ApplicantIncome': float(applicant_income),
        'CoapplicantIncome': float(coapplicant_income),
        'LoanAmount': float(loan_amount),
        'LoanAmountTerm': float(loan_amount_term),
        'Credit_History': int(credit_history),
        'Dependents': int(dependents)
    }

    # Build DataFrame in the exact training column order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=FEATURE_ORDER, fill_value=0)

    # Prediction
    try:
        prediction = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")

    # Display result
    if prediction is not None:
        if prediction == 1:
            st.success("Loan is likely to be Approved!")
        else:
            st.error("Loan is likely to be Rejected.")

# Optional: Show input data (for debugging)
if 'input_df' in locals() and input_df is not None:
    if st.checkbox("Show input data (for debugging)"):
        st.write(input_df)