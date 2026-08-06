from pathlib import Path
from urllib.parse import urlparse
import re
import string

import joblib
from scipy.sparse import hstack


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "naive_bayes_model.joblib"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "naive_bayes_vectorizer.joblib"
)


class NaiveBayesPredictor:
    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None

    def load(self) -> None:
        """Load the trained model and fitted vectorizer."""

        self.model = joblib.load(
            MODEL_PATH
        )

        self.vectorizer = joblib.load(
            VECTORIZER_PATH
        )

    @staticmethod
    def get_sender_domain(sender: str) -> str:
        """Extract the domain from the sender email."""

        sender = sender.strip().lower()

        if "@" not in sender:
            return ""

        return sender.rsplit("@", 1)[1]

    @staticmethod
    def extract_urls(text: str) -> list[str]:
        """Extract URLs from the subject and email body."""

        return re.findall(
            r"https?://\S+|www\.\S+",
            text,
        )

    @staticmethod
    def count_subdomains(url: str) -> int:
        """Count the number of subdomains in a URL."""

        hostname = urlparse(url).hostname

        if not hostname:
            return 0

        return max(
            len(hostname.split(".")) - 2,
            0,
        )

    def prepare_input(
        self,
        sender: str,
        subject: str,
        body: str,
        attachment_count: int = 0,
    ):
        """
        Create the input expected by the trained model.

        The model expects:
        20,000 TF-IDF features
        + 11 engineered features
        = 20,011 total features.
        """

        if self.vectorizer is None:
            raise RuntimeError(
                "Naive Bayes vectorizer has not been loaded."
            )

        if self.model is None:
            raise RuntimeError(
                "Naive Bayes model has not been loaded."
            )

        sender = sender.strip()
        subject = subject.strip()
        body = body.strip()

        sender_domain = self.get_sender_domain(
            sender
        )

        # Match how the text was combined during training.
        email_text = f"{subject} {body}".strip()

        # Convert the email text into TF-IDF features.
        text_vector = self.vectorizer.transform(
            [email_text]
        )

        urls = self.extract_urls(
            email_text
        )

        url_lengths = [
            len(url)
            for url in urls
        ]

        subdomain_counts = [
            self.count_subdomains(url)
            for url in urls
        ]

        url_count = len(urls)

        url_length_max = (
            max(url_lengths)
            if url_lengths
            else 0
        )

        url_length_avg = (
            sum(url_lengths) / len(url_lengths)
            if url_lengths
            else 0
        )

        url_subdom_max = (
            max(subdomain_counts)
            if subdomain_counts
            else 0
        )

        url_subdom_avg = (
            sum(subdomain_counts)
            / len(subdomain_counts)
            if subdomain_counts
            else 0
        )

        punctuation_count = sum(
            character in string.punctuation
            for character in email_text
        )

        # This order must exactly match the training notebook:
        #
        # 1. sender_domain_length
        # 2. subject_length
        # 3. body_length
        # 4. email_length
        # 5. punctuation_count
        # 6. url_count
        # 7. url_length_max
        # 8. url_length_avg
        # 9. url_subdom_max
        # 10. url_subdom_avg
        # 11. attachment_count

        numeric_values = [[
            len(sender_domain),
            len(subject),
            len(body),
            len(email_text),
            punctuation_count,
            url_count,
            url_length_max,
            url_length_avg,
            url_subdom_max,
            url_subdom_avg,
            attachment_count,
        ]]

        model_input = hstack(
            [
                text_vector,
                numeric_values,
            ]
        ).tocsr()

        # Safety check so the error is easier to understand.
        expected_features = (
            self.model.n_features_in_
        )

        actual_features = (
            model_input.shape[1]
        )

        if actual_features != expected_features:
            raise ValueError(
                "Naive Bayes input feature mismatch: "
                f"created {actual_features} features, "
                f"but the model expects "
                f"{expected_features}."
            )

        return model_input

    def predict(
        self,
        sender: str,
        subject: str,
        body: str,
        attachment_count: int = 0,
    ) -> dict:
        """Predict whether the email is phishing."""

        if self.model is None:
            raise RuntimeError(
                "Naive Bayes model has not been loaded."
            )

        model_input = self.prepare_input(
            sender=sender,
            subject=subject,
            body=body,
            attachment_count=attachment_count,
        )

        raw_prediction = self.model.predict(
            model_input
        )[0]

        probabilities = self.model.predict_proba(
            model_input
        )[0]

        prediction_index = int(
            raw_prediction
        )

        is_phishing = (
            prediction_index == 1
        )

        return {
            "label": (
                "Phishing"
                if is_phishing
                else "Legitimate"
            ),
            "is_phishing": is_phishing,
            "confidence": float(
                probabilities[
                    prediction_index
                ]
            ),
            "legitimate_probability": float(
                probabilities[0]
            ),
            "phishing_probability": float(
                probabilities[1]
            ),
        }