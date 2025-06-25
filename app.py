
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()


model_path = current_dir / ".." / "notebooks" / "fraud_detection_pipeline.pkl"

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(
        f"Error: Model file not found at {model_path}. Please check the path.")
    # Optionally, stop the app if the model is crucial
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()


# model = joblib.load(Path("fraud_detection_pipeline.pkl"))


st.title("Fraud Detection Prediction App")

st.markdown("Please enter the transaction details and use hte predict button")

st.divider()

transaction_type = st.selectbox(
    "transaction Type", ["PAYMENT", "TRANSFER", "CHAS_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.8, value=1000.0)
oldbalanceOrg = st.number_input(
    "Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input(
    "New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input(
    "Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input(
    "New Balance (Receiver)", min_value=0.0, value=0.0)


if st.button("Predict"):
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
    }])

    prediction = model.predict(input_data)[0]

    st.subheader(f"[+] Prediction : '{int(prediction)}")

    if prediction == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks like is not fraud")
