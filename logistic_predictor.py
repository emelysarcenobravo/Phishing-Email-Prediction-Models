from pathlib import Path
from urllib.parse import urlparse
import __main__
import re
import string

import joblib
import pandas as pd
from spellchecker import SpellChecker

from custom_transformers import MultiLabelBinarizerTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "logistic_regression_pipeline.joblib"
)


class LogisticPredictor:
    def __init__(self) -> None:
        self.model = None
        self.spell_checker = SpellChecker()

    def load(self) -> None:
        # Required because the pipeline was saved from a notebook.
        __main__.MultiLabelBinarizerTransformer = (
            MultiLabelBinarizerTransformer
        )

        self.model = joblib.load(MODEL_PATH)

    @staticmethod
    def get_domain(email_address: str) -> str:
        email_address = email_address.strip().lower()

        if "@" not in email_address:
            return ""

        return email_address.rsplit("@", 1)[1]

    @staticmethod
    def extract_urls(text: str) -> list[str]:
        return re.findall(
            r'https?://[^\s<>"]+',
            text,
        )

    @staticmethod
    def count_url_subdomains(url: str) -> int:
        hostname = urlparse(url).hostname

        if not hostname:
            return 0

        return max(
            len(hostname.split(".")) - 2,
            0,
        )

    def get_spelling_features(
        self,
        text: str,
    ) -> dict:
        words = re.findall(
            r"\b[a-zA-Z]+\b",
            str(text).lower(),
        )

        if not words:
            return {
                "misspelled_count": 0,
                "misspelled_ratio": 0.0,
            }

        misspelled_words = (
            self.spell_checker.unknown(words)
        )

        return {
            "misspelled_count": len(
                misspelled_words
            ),
            "misspelled_ratio": (
                len(misspelled_words)
                / len(words)
            ),
        }

    def prepare_input(
        self,
        sender: str,
        receiver: str,
        subject: str,
        body: str,
        attachment_count: int = 0,
    ) -> pd.DataFrame:
        sender_domain = self.get_domain(sender)
        receiver_domain = self.get_domain(receiver)

        urls = self.extract_urls(body)

        url_lengths = [
            len(url)
            for url in urls
        ]

        url_subdomains = [
            self.count_url_subdomains(url)
            for url in urls
        ]

        subject_spelling = (
            self.get_spelling_features(subject)
        )

        body_spelling = (
            self.get_spelling_features(body)
        )

        input_row = {
            "subject": subject,
            "body": body,
            "sender_domain": sender_domain,
            "receiver_domain": receiver_domain,
            "urls": " ".join(urls),

            "subject_length": len(subject),
            "body_length": len(body),
            "email_length": (
                len(subject) + len(body)
            ),
            "sender_domain_length": len(
                sender_domain
            ),

            "url_count": len(urls),

            "url_length_max": (
                max(url_lengths)
                if url_lengths
                else 0
            ),

            "url_length_avg": (
                sum(url_lengths)
                / len(url_lengths)
                if url_lengths
                else 0
            ),

            "url_subdom_max": (
                max(url_subdomains)
                if url_subdomains
                else 0
            ),

            "url_subdom_avg": (
                sum(url_subdomains)
                / len(url_subdomains)
                if url_subdomains
                else 0
            ),

            "attachment_count": attachment_count,

            "punctuation_count": sum(
                character in string.punctuation
                for character in body
            ),

            "subject_misspelled_count": (
                subject_spelling[
                    "misspelled_count"
                ]
            ),

            "subject_misspelled_ratio": (
                subject_spelling[
                    "misspelled_ratio"
                ]
            ),

            "body_misspelled_count": (
                body_spelling[
                    "misspelled_count"
                ]
            ),

            "body_misspelled_ratio": (
                body_spelling[
                    "misspelled_ratio"
                ]
            ),

            "source": "user_input",
            "has_attachments": (
                attachment_count > 0
            ),

            "content_types": [
                "text/plain"
            ],

            "attachment_types": [""],
        }

        return pd.DataFrame([input_row])

    def predict(
        self,
        sender: str,
        receiver: str,
        subject: str,
        body: str,
        attachment_count: int = 0,
    ) -> dict:
        if self.model is None:
            raise RuntimeError(
                "Logistic Regression model "
                "has not been loaded."
            )

        model_input = self.prepare_input(
            sender=sender,
            receiver=receiver,
            subject=subject,
            body=body,
            attachment_count=attachment_count,
        )

        prediction = self.model.predict(
            model_input
        )[0]

        probabilities = self.model.predict_proba(
            model_input
        )[0]

        is_phishing = int(prediction) == 1

        return {
            "label": (
                "Phishing"
                if is_phishing
                else "Legitimate"
            ),
            "is_phishing": is_phishing,
            "confidence": float(
                probabilities[int(prediction)]
            ),
            "legitimate_probability": float(
                probabilities[0]
            ),
            "phishing_probability": float(
                probabilities[1]
            ),
            "model_input": model_input,
        }