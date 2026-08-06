import pandas as pd
import streamlit as st


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧",
    layout="wide",
)


# ==================================================
# Final evaluation metrics
# ==================================================

MODEL_METRICS = {
    "Logistic Regression": {
        "Accuracy": 0.9928,
        "Precision": 0.9944,
        "Recall": 0.9931,
        "F1 Score": 0.9938,
        "Rank": 1,
        "Status": "🏆 Best Overall",
    },
    "BERT": {
        "Accuracy": 0.9800,
        "Precision": 0.9900,
        "Recall": 0.9900,
        "F1 Score": 0.9900,
        "Rank": 2,
        "Status": "⭐ Excellent",
    },
    "Multinomial Naive Bayes": {
        "Accuracy": 0.7469,
        "Precision": 0.6987,
        "Recall": 0.7513,
        "F1 Score": 0.7240,
        "Rank": 3,
        "Status": "Baseline",
    },
}


# ==================================================
# Cached model loaders
# ==================================================

@st.cache_resource
def load_logistic_predictor():
    from logistic_predictor import LogisticPredictor

    predictor = LogisticPredictor()
    predictor.load()
    return predictor


@st.cache_resource
def load_naive_bayes_predictor():
    from naive_bayes_predictor import NaiveBayesPredictor

    predictor = NaiveBayesPredictor()
    predictor.load()
    return predictor


@st.cache_resource
def load_bert_predictor():
    from bert_predictor import BertPredictor

    predictor = BertPredictor()
    predictor.load()
    return predictor


# ==================================================
# Helper functions
# ==================================================

def format_percentage(value: float) -> str:
    """Convert a decimal metric into a percentage string."""
    return f"{value:.2%}"


def run_selected_model(
    selected_model: str,
    sender: str,
    receiver: str,
    subject: str,
    body: str,
    attachment_count: int,
) -> dict:
    """Load and run the model selected by the user."""

    if selected_model == "Logistic Regression":
        predictor = load_logistic_predictor()

        return predictor.predict(
            sender=sender,
            receiver=receiver,
            subject=subject,
            body=body,
            attachment_count=attachment_count,
        )

    if selected_model == "Multinomial Naive Bayes":
        predictor = load_naive_bayes_predictor()

        return predictor.predict(
            sender=sender,
            subject=subject,
            body=body,
            attachment_count=attachment_count,
        )

    predictor = load_bert_predictor()

    return predictor.predict(
        subject=subject,
        body=body,
    )


def create_leaderboard() -> pd.DataFrame:
    """Create the final model leaderboard."""

    leaderboard_rows = []

    for model_name, metrics in MODEL_METRICS.items():
        leaderboard_rows.append(
            {
                "Rank": metrics["Rank"],
                "Model": model_name,
                "Accuracy": format_percentage(
                    metrics["Accuracy"]
                ),
                "Precision": format_percentage(
                    metrics["Precision"]
                ),
                "Recall": format_percentage(
                    metrics["Recall"]
                ),
                "F1 Score": format_percentage(
                    metrics["F1 Score"]
                ),
                "Result": metrics["Status"],
            }
        )

    return (
        pd.DataFrame(leaderboard_rows)
        .sort_values("Rank")
        .reset_index(drop=True)
    )


# ==================================================
# Header
# ==================================================

st.title("📧 Phishing Email Detection")

st.write(
    "Compare three machine-learning models and use the selected "
    "model to classify an email as phishing or legitimate."
)

st.markdown(
    "**Dataset:** "
    "[Hugging Face – Phishing Email Dataset]"
    "(https://huggingface.co/datasets/"
    "simlab-vs/meajor_cleaned_preprocessed)"
)

st.divider()


# ==================================================
# Model dashboard
# ==================================================

st.subheader("Model Dashboard")

selected_model = st.selectbox(
    "Select a model",
    [
        "Logistic Regression",
        "BERT",
        "Multinomial Naive Bayes",
    ],
)

selected_metrics = MODEL_METRICS[selected_model]

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

metric_col1.metric(
    "Accuracy",
    format_percentage(selected_metrics["Accuracy"]),
)

metric_col2.metric(
    "Precision",
    format_percentage(selected_metrics["Precision"]),
)

metric_col3.metric(
    "Recall",
    format_percentage(selected_metrics["Recall"]),
)

metric_col4.metric(
    "F1 Score",
    format_percentage(selected_metrics["F1 Score"]),
)

if selected_metrics["Rank"] == 1:
    st.success(
        f"{selected_metrics['Status']}: "
        f"{selected_model} achieved the strongest overall results."
    )
else:
    st.info(
        f"Current ranking: #{selected_metrics['Rank']} — "
        f"{selected_metrics['Status']}"
    )

st.divider()


# ==================================================
# Email form
# ==================================================

st.subheader("Analyze an Email")

st.write(
    "A demo phishing-style email is provided below. "
    "You can analyze it immediately or replace it with your own email."
)

with st.form(key="email_form"):

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
            "Click the link below to verify your information "
            "immediately.\n\n"
            "https://login.account-verification.com/verify"
        ),
        height=240,
    )

    attachment_count = st.number_input(
        "Number of Attachments",
        min_value=0,
        value=0,
        step=1,
    )

    submitted = st.form_submit_button(
        "Analyze Email",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# Prediction
# ==================================================

if submitted:

    if not sender.strip():
        st.warning(
            "Please enter the sender's email address."
        )

    elif "@" not in sender:
        st.warning(
            "Please enter a valid sender email address."
        )

    elif receiver.strip() and "@" not in receiver:
        st.warning(
            "Please enter a valid receiver email address."
        )

    elif not subject.strip() and not body.strip():
        st.warning(
            "Please enter either the email subject or body."
        )

    else:
        try:
            with st.spinner(
                f"Loading and running {selected_model}..."
            ):
                result = run_selected_model(
                    selected_model=selected_model,
                    sender=sender,
                    receiver=receiver,
                    subject=subject,
                    body=body,
                    attachment_count=int(
                        attachment_count
                    ),
                )

            st.divider()
            st.subheader("Prediction Result")

            if result["is_phishing"]:
                st.error(
                    "⚠️ This email was classified as phishing."
                )
            else:
                st.success(
                    "✅ This email was classified as legitimate."
                )

            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )

            result_col1.metric(
                "Model",
                selected_model,
            )

            result_col2.metric(
                "Prediction",
                result["label"],
            )

            result_col3.metric(
                "Confidence",
                format_percentage(
                    result["confidence"]
                ),
            )

            st.subheader("Class Probabilities")

            legitimate_probability = result[
                "legitimate_probability"
            ]

            phishing_probability = result[
                "phishing_probability"
            ]

            probability_table = pd.DataFrame(
                {
                    "Class": [
                        "Legitimate",
                        "Phishing",
                    ],
                    "Probability": [
                        format_percentage(
                            legitimate_probability
                        ),
                        format_percentage(
                            phishing_probability
                        ),
                    ],
                }
            )

            st.dataframe(
                probability_table,
                hide_index=True,
                use_container_width=True,
            )

            st.progress(
                phishing_probability,
                text=(
                    "Phishing probability: "
                    f"{phishing_probability:.2%}"
                ),
            )

            if "model_input" in result:
                with st.expander(
                    "View engineered input sent to the model"
                ):
                    st.dataframe(
                        result["model_input"],
                        hide_index=True,
                        use_container_width=True,
                    )

        except FileNotFoundError as error:
            st.error(
                "A required model file could not be found."
            )
            st.exception(error)

        except Exception as error:
            st.error(
                f"{selected_model} could not complete "
                "the prediction."
            )
            st.exception(error)


# ==================================================
# Leaderboard
# ==================================================

st.divider()
st.subheader("Model Leaderboard")

leaderboard = create_leaderboard()

st.dataframe(
    leaderboard,
    hide_index=True,
    use_container_width=True,
)

winner = leaderboard.iloc[0]

st.success(
    f"🏆 Overall Winner: {winner['Model']}"
)

st.write(
    f"{winner['Model']} ranked first with an accuracy of "
    f"{winner['Accuracy']} and an F1 score of "
    f"{winner['F1 Score']}."
)

st.caption(
    "The leaderboard is ranked using the final model evaluation results."
)
