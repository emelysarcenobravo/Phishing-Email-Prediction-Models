from pathlib import Path
import __main__

import joblib
import pandas as pd
import streamlit as st

from custom_transformers import MultiLabelBinarizerTransformer


st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "logistic_regression_pipeline.joblib"
)


@st.cache_resource
def load_logistic_model():
    __main__.MultiLabelBinarizerTransformer = (
        MultiLabelBinarizerTransformer
    )

    return joblib.load(MODEL_PATH)


st.title("📧 Phishing Email Detection")

st.write("Testing the Logistic Regression model first.")

try:
    with st.spinner("Loading Logistic Regression model..."):
        model = load_logistic_model()

    st.success("Logistic Regression loaded successfully!")

except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()


subject = st.text_input(
    "Subject",
    value="Urgent: Verify your account immediately",
)

body = st.text_area(
    "Email Body",
    value=(
        "Your account has been suspended. "
        "Click the link below to verify your information immediately."
    ),
    height=200,
)

st.info(
    "The model is loaded. Prediction input will be connected next."
)