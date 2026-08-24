import __main__

import joblib
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from custom_transformers import MultiLabelBinarizerTransformer


__main__.MultiLabelBinarizerTransformer = (
    MultiLabelBinarizerTransformer
)

# Logistic Regression
logistic_model = joblib.load(
    "models/logistic_regression_pipeline.joblib"
)
print("Logistic Regression loaded successfully")


# Naive Bayes
naive_bayes_model = joblib.load(
    "models/naive_bayes_model.joblib"
)

naive_bayes_vectorizer = joblib.load(
    "models/naive_bayes_vectorizer.joblib"
)

print("Naive Bayes model and vectorizer loaded successfully")


# BERT
bert_path = "models/distilbert_phishing_model"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    bert_path
)

print("BERT model and tokenizer loaded successfully")