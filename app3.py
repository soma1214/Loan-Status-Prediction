# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model artifacts
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('target_mapping.pkl', 'rb') as f:
    target_mapping = pickle.load(f)

# invert target mapping for display: int -> original label (e.g., 'Y' or 'N')
inv_target = {v: k for k, v in target_mapping.items()}

st.title("🏦 Loan Status Predictor")
st.write("Fill the form and click **Predict**. (Encodings are taken from training-time mappings.)")

# Build inputs dynamically using model_columns (keeps exact names used at training)
input_vals = {}
for col in model_columns:
    display_name = col.replace('_', ' ').title()

    # If this column was label-encoded at training time, show original string options
    if col in encoders:
        # order options by the encoding integer so we present them in the same order
        opts = sorted(list(encoders[col].keys()), key=lambda x: encoders[col][x])
        choice = st.selectbox(display_name, opts)
        input_vals[col] = encoders[col][choice]

    else:
        # numeric inputs (common known numeric columns)
        if col == 'Dependents':
            v = st.number_input(display_name, min_value=0, step=1, value=0)
            input_vals[col] = int(v)
        elif col == 'Credit_History':
            v = st.selectbox(display_name, [1, 0])
            input_vals[col] = int(v)
        elif col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
            v = st.number_input(display_name, min_value=0.0, value=0.0, format="%.2f")
            # Loan_Amount_Term typically an integer (months), but float is fine — we'll cast later
            input_vals[col] = float(v)
        else:
            # fallback numeric
            v = st.number_input(display_name, value=0.0)
            input_vals[col] = float(v)

# Prediction button
if st.button("Predict"):
    # ensure ordering exactly same as training
    input_df = pd.DataFrame([input_vals], columns=model_columns)

    # cast known integer columns
    if 'Loan_Amount_Term' in input_df.columns:
        input_df['Loan_Amount_Term'] = pd.to_numeric(input_df['Loan_Amount_Term']).astype(float)
    if 'Dependents' in input_df.columns:
        input_df['Dependents'] = pd.to_numeric(input_df['Dependents']).astype(int)
    if 'Credit_History' in input_df.columns:
        input_df['Credit_History'] = pd.to_numeric(input_df['Credit_History']).astype(int)

    # predict
    pred = model.predict(input_df)[0]
    pred_label = inv_target.get(int(pred), str(pred))

    if int(pred) == 1:
        st.success(f"Prediction: {pred_label} — **Loan likely Approved**")
    else:
        st.error(f"Prediction: {pred_label} — **Loan likely Rejected**")
