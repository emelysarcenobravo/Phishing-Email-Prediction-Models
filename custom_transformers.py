from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        # Expects a pandas Series or 1D array of iterables
        self.mlb.fit(X.squeeze())
        return self

    def transform(self, X):
        return self.mlb.transform(X.squeeze())

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_