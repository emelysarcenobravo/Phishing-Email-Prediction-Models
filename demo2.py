from pathlib import Path
from urllib.parse import urlparse
import __main__
import re
import string

import joblib
import pandas as pd
import streamlit as st
from spellchecker import SpellChecker

from custom_transformers import MultiLabelBinarizerTransformer


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧",
    layout="wide",
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "logistic_regression_pipeline.joblib"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_logistic_model():
    """
    Load the saved Logistic Regression pipeline.

    The __main__ assignment is required because the pipeline
    was originally saved from a notebook.
    """

    __main__.MultiLabelBinarizerTransformer = (
        MultiLabelBinarizerTransformer
    )

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_spell_checker():
    return SpellChecker()


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def get_domain(email_address: str) -> str:
    """Extract the domain from an email address."""

    email_address = email_address.strip().lower()

    if "@" not in email_address:
        return ""

    return email_address.rsplit("@", 1)[1]


def extract_urls(text: str) -> list[str]:
    """Extract URLs from the email body."""

    pattern = r"https?://[^\s<>\"]+"

    return re.findall(pattern, text)


def count_url_subdomains(url: str) -> int:
    """Count subdomains in a URL."""

    hostname = urlparse(url).hostname

    if not hostname:
        return 0

    parts = hostname.split(".")

    # Example:
    # login.security.example.com
    # has two subdomains: login and security
    return max(len(parts) - 2, 0)


def get_spelling_features(text: str, spell: SpellChecker) -> dict:
    """
    Reproduce the spelling feature logic from the
    Logistic Regression notebook.
    """

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        str(text).lower(),
    )

    if not words:
        return {
            "misspelled_count": 0,
            "misspelled_ratio": 0.0,
        }

    misspelled_words = spell.unknown(words)

    return {
        "misspelled_count": len(misspelled_words),
        "misspelled_ratio": (
            len(misspelled_words) / len(words)
        ),
    }


def prepare_logistic_input(
    sender: str,
    receiver: str,
    subject: str,
    body: str,
) -> pd.DataFrame:
    """
    Build one input row with the columns expected by the
    saved Logistic Regression preprocessing pipeline.
    """

    spell = load_spell_checker()

    sender_domain = get_domain(sender)
    receiver_domain = get_domain(receiver)

    urls = extract_urls(body)
    url_lengths = [len(url) for url in urls]
    url_subdomain_counts = [
        count_url_subdomains(url)
        for url in urls
    ]

    subject_spelling = get_spelling_features(
        subject,
        spell,
    )

    body_spelling = get_spelling_features(
        body,
        spell,
    )

    input_row = {
        # Text features
        "subject": subject,
        "body": body,

        # Domain and URL text features
        "sender_domain": sender_domain,
        "receiver_domain": receiver_domain,
        "urls": " ".join(urls),

        # Numerical features
        "subject_length": len(subject),
        "body_length": len(body),

        # The training notebook used subject length + body length.
        "email_length": len(subject) + len(body),

        "sender_domain_length": len(sender_domain),

        "url_count": len(urls),

        "url_length_max": (
            max(url_lengths)
            if url_lengths
            else 0
        ),

        "url_length_avg": (
            sum(url_lengths) / len(url_lengths)
            if url_lengths
            else 0
        ),

        "url_subdom_max": (
            max(url_subdomain_counts)
            if url_subdomain_counts
            else 0
        ),

        "url_subdom_avg": (
            sum(url_subdomain_counts)
            / len(url_subdomain_counts)
            if url_subdomain_counts
            else 0
        ),

        # This first version does not upload attachments.
        "attachment_count": 0,

        # The notebook calculated punctuation from the body.
        "punctuation_count": sum(
            character in string.punctuation
            for character in body
        ),

        # Spelling features
        "subject_misspelled_count": (
            subject_spelling["misspelled_count"]
        ),

        "subject_misspelled_ratio": (
            subject_spelling["misspelled_ratio"]
        ),

        "body_misspelled_count": (
            body_spelling["misspelled_count"]
        ),

        "body_misspelled_ratio": (
            body_spelling["misspelled_ratio"]
        ),

        # Categorical features
        #
        # The source of a user-submitted email is unknown.
        # OneHotEncoder(handle_unknown="ignore") safely handles it.
        "source": "user_input",

        "has_attachments": False,

        # Multi-label columns must contain lists.
        "content_types": ["text/plain"],
        "attachment_types": [""],
    }

    return pd.DataFrame([input_row])


def normalize_label(prediction) -> tuple[str, bool]:
    """
    Convert the model's output into a readable result.

    The dataset uses 1 for phishing and 0 for legitimate.
    """

    prediction_text = str(prediction).strip().lower()

    phishing_values = {
        "1",
        "1.0",
        "phishing",
        "spam",
    }

    is_phishing = prediction_text in phishing_values

    if is_phishing:
        return "Phishing", True

    return "Legitimate", False


# --------------------------------------------------
# Application
# --------------------------------------------------

st.title("📧 Phishing Email Detection")

st.write(
    "Enter an email below and analyze it using "
    "the Logistic Regression model."
)


try:
    with st.spinner(
        "Loading Logistic Regression model..."
    ):
        model = load_logistic_model()

    st.success(
        "Logistic Regression loaded successfully!"
    )

except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()


st.divider()

st.subheader("Analyze an Email")

with st.form("email_form"):

    sender = st.text_input(
        "Sender Email",
        value="security@account-verification.com",
    )

    receiver = st.text_input(
        "Receiver Email",
        value="student@example.com",
    )

    subject = st.text_input(
        "Subject",
        value="Urgent: Verify your account immediately",
    )

    body = st.text_area(
        "Email Body",
        value=(
            "Your account has been suspended. "
            "Click the link below to verify your "
            "information immediately.\n\n"
            "https://login.account-verification.com/verify"
        ),
        height=240,
    )

    submitted = st.form_submit_button(
        "Analyze Email",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    if not subject.strip() and not body.strip():
        st.warning(
            "Please enter an email subject or body."
        )

    else:
        try:
            model_input = prepare_logistic_input(
                sender=sender,
                receiver=receiver,
                subject=subject,
                body=body,
            )

            with st.spinner("Analyzing email..."):
                raw_prediction = model.predict(
                    model_input
                )[0]

                label, is_phishing = normalize_label(
                    raw_prediction
                )

                confidence = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        model_input
                    )[0]

                    confidence = float(
                        max(probabilities)
                    )

            st.divider()
            st.subheader("Prediction Result")

            if is_phishing:
                st.error(
                    "⚠️ This email was classified as phishing."
                )
            else:
                st.success(
                    "✅ This email was classified as legitimate."
                )

            if confidence is not None:
                st.metric(
                    "Model Confidence",
                    f"{confidence:.2%}",
                )

            with st.expander(
                "View input features sent to the model"
            ):
                st.dataframe(
                    model_input,
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)