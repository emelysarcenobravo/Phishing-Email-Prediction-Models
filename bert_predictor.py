from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "distilbert_phishing_model"
)


class BertPredictor:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_PATH
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_PATH)
        )

        self.model.eval()

    def prepare_input(
        self,
        subject: str,
        body: str,
    ) -> dict:
        if self.tokenizer is None:
            raise RuntimeError(
                "BERT tokenizer has not been loaded."
            )

        # Match the training notebook.
        email_text = subject + " " + body

        return self.tokenizer(
            email_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

    def predict(
        self,
        subject: str,
        body: str,
    ) -> dict:
        if self.model is None:
            raise RuntimeError(
                "BERT model has not been loaded."
            )

        encoded_input = self.prepare_input(
            subject=subject,
            body=body,
        )

        with torch.no_grad():
            output = self.model(
                **encoded_input
            )

        probabilities = torch.softmax(
            output.logits,
            dim=1,
        )[0]

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        is_phishing = prediction == 1

        return {
            "label": (
                "Phishing"
                if is_phishing
                else "Legitimate"
            ),
            "is_phishing": is_phishing,
            "confidence": float(
                probabilities[prediction].item()
            ),
            "legitimate_probability": float(
                probabilities[0].item()
            ),
            "phishing_probability": float(
                probabilities[1].item()
            ),
        }